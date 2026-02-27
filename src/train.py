import os
import time
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader

from distributed.sync_schemes import build_sync_controller
from metrics.perf import count_params
from metrics.stat import eval_f1
from models.graphsage import GraphSAGE
from runtime.ddp import ddp_setup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=os.environ.get("PYG_DATA_ROOT", os.path.expandvars("$SCRATCH/pyg_datasets")))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=[25, 10])
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sync_every", type=int, default=1)
    parser.add_argument(
        "--sync_scheme",
        type=str,
        default="deferred_allreduce",
        choices=["deferred_allreduce", "local_sgd"],
        help="Synchronization strategy for distributed training.",
    )
    parser.add_argument("--max_steps", type=int, default=None,
                    help="Total global optimizer steps (overrides epochs if set)")
    args = parser.parse_args()

    is_distributed, rank, local_rank, world_size = ddp_setup()
    is_rank0 = (rank == 0)

    if args.max_steps is not None:
        if is_distributed:
            assert args.max_steps % world_size == 0, \
                "--max_steps must be divisible by world_size"
            max_local_steps = args.max_steps // world_size
        else:
            max_local_steps = args.max_steps
    else:
        max_local_steps = None

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if is_rank0:
        print(f"Dataset root: {args.dataset_root}")
    dataset = Reddit(root=args.dataset_root)
    data = dataset[0]

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    val_idx   = data.val_mask.nonzero(as_tuple=False).view(-1)

    # Partition across ranks for training
    if is_distributed:
        train_idx_rank = train_idx[rank::world_size]
    else:
        train_idx_rank = train_idx

    num_features = dataset.num_features
    num_classes = dataset.num_classes

    if is_rank0:
        print(dataset)
        print(data)

    sync_every = args.sync_every
    sync_controller = build_sync_controller(
        scheme=args.sync_scheme,
        sync_every=sync_every,
        is_distributed=is_distributed,
    )
    # TODO(local-sgd): add an optional LocalSGD mode that performs k local optimizer
    # steps and periodically averages model parameters across ranks (instead of using
    # deferred gradient allreduce via DDP.no_sync).

    # Neighbor loaders for train/val/test
    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx_rank,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # start with 0 for stability on HPC; increase later
    )

    if is_rank0:
        val_loader = NeighborLoader(
            data,
            input_nodes=val_idx,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        val_loader = None

    model = GraphSAGE(
        in_channels=num_features,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    BYTES_TO_MB = 1.0 / (1024 ** 2)
    base_model = model.module if hasattr(model, "module") else model
    param_count = count_params(base_model)
    param_mb_fp16 = param_count * 2 *  BYTES_TO_MB
    param_mb_fp32 = param_count * 4 * BYTES_TO_MB
    param_mb_fp64 = param_count * 8 * BYTES_TO_MB
    if is_rank0:
        print(f"LOG,param_count={param_count},"
            f"param_mb_fp16={param_mb_fp16:.3f},param_mb_fp32={param_mb_fp32:.3f},param_mb_fp64={param_mb_fp64:.3f}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    comm_mb_cum_fp16 = 0
    comm_mb_cum_fp32 = 0
    comm_mb_cum_fp64 = 0

    best_micro_f1 = 0.0
    best_macro_f1 = 0.0
    global_step = 0
    epoch = 0
    seed_nodes_cum = 0
    while True:
        t0 = time.time()
        epoch += 1
        model.train()
        seed_nodes_epoch = 0
        steps = 0
        total_loss = 0.0
        sync_controller.start_epoch(optimizer)
        stop_requested = False
        for batch in train_loader:
            batch = batch.to(device)

            with sync_controller.backward_context(model):
                out = model(batch.x, batch.edge_index)
                y = batch.y[: batch.batch_size]
                out = out[: batch.batch_size]
                seed_nodes_epoch += batch.batch_size
                loss = F.cross_entropy(out, y)
                sync_controller.backward(loss)

            total_loss += loss.item()
            steps += 1

            if sync_controller.maybe_step(model, optimizer):
                global_step += 1
                if max_local_steps is not None and global_step >= max_local_steps:
                    stop_requested = True
                    break

        if sync_controller.finalize_epoch(model, optimizer, stop_requested=stop_requested):
            global_step += 1

        syncs_this_epoch = sync_controller.syncs_this_epoch
        seed_nodes_epoch_tensor = torch.tensor(seed_nodes_epoch, device=device, dtype=torch.long)
        if is_distributed:
            dist.all_reduce(seed_nodes_epoch_tensor, op=dist.ReduceOp.SUM)

        seed_nodes_epoch_global = int(seed_nodes_epoch_tensor.item())
        if is_rank0:
            seed_nodes_cum += seed_nodes_epoch_global
        epoch_time = time.time() - t0
        done = (max_local_steps is not None and global_step >= max_local_steps)

        avg_loss = total_loss / max(steps, 1)

        comm_mb_epoch_fp16 = syncs_this_epoch * param_mb_fp16
        comm_mb_epoch_fp32 = syncs_this_epoch * param_mb_fp32
        comm_mb_epoch_fp64 = syncs_this_epoch * param_mb_fp64

        commMB_per_step_fp16 = (comm_mb_epoch_fp16 / steps) if steps else 0.0
        commMB_per_step_fp32 = (comm_mb_epoch_fp32 / steps) if steps else 0.0
        commMB_per_step_fp64 = (comm_mb_epoch_fp64 / steps) if steps else 0.0

        comm_mb_cum_fp16 += comm_mb_epoch_fp16
        comm_mb_cum_fp32 += comm_mb_epoch_fp32
        comm_mb_cum_fp64 += comm_mb_epoch_fp64

        if is_rank0:
            eval_model = model.module if hasattr(model, "module") else model
            micro_f1, macro_f1 = eval_f1(eval_model, val_loader, device, num_classes)
        else:
            micro_f1, macro_f1 = 0.0, 0.0
        best_micro_f1 = max(best_micro_f1, micro_f1)
        best_macro_f1 = max(best_macro_f1, macro_f1)
        if is_rank0:
            effective_global_steps = global_step * world_size
            print(
                f"LOG,"
                f"epoch={epoch},"
                f"loss={avg_loss:.6f},"
                f"microF1={micro_f1:.6f},"
                f"macroF1={macro_f1:.6f},"
                f"best_microF1={best_micro_f1:.6f},"
                f"best_macroF1={best_macro_f1:.6f},"
                f"local_steps={global_step},"
                f"epoch_steps={steps},"
                f"effective_global_steps={effective_global_steps},"
                f"seed_nodes_epoch_global={seed_nodes_epoch_global},"
                f"seed_nodes_cum_global={seed_nodes_cum},"
                f"target_global_steps={args.max_steps if args.max_steps is not None else 'NA'},"
                f"world_size={world_size},"
                f"time={epoch_time:.6f},"
                f"sync_every={sync_every},"
                f"syncs={syncs_this_epoch},"
                f"commMB_per_step_fp16={commMB_per_step_fp16:.3f},"
                f"commMB_per_step_fp32={commMB_per_step_fp32:.3f},"
                f"commMB_per_step_fp64={commMB_per_step_fp64:.3f},"
                f"commMB_epoch_fp16={comm_mb_epoch_fp16:.3f},"
                f"commMB_cum_fp16={comm_mb_cum_fp16:.3f},"
                f"commMB_epoch_fp32={comm_mb_epoch_fp32:.3f},"
                f"commMB_cum_fp32={comm_mb_cum_fp32:.3f},"
                f"commMB_epoch_fp64={comm_mb_epoch_fp64:.3f},"
                f"commMB_cum_fp64={comm_mb_cum_fp64:.3f}",
                flush=True
            )
        if done:
            break

    if is_rank0:
        print("Done.")

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
