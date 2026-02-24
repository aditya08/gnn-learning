import os

import torch
import torch.distributed as dist

def ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

        rank = dist.get_rank()
        dist.barrier()  # Ensure all processes have initialized before proceeding
        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1