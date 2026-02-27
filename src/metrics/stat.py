import torch
"""
Evaluate F1 scores (micro and macro) for a multi-class classification model.

This function computes both micro-F1 and macro-F1 scores by building a confusion
matrix across all batches in the data loader. It operates in evaluation mode with
gradients disabled.

Args:
    model: A PyTorch model that takes node features and edge indices as input
        and outputs class predictions.
    loader: A PyTorch DataLoader containing batches with attributes:
        - x: Node features
        - edge_index: Graph connectivity
        - y: Ground truth labels
        - batch_size: Number of samples in the batch
    device: The device (CPU/GPU) on which to perform computations.
    num_classes (int): The total number of classes in the classification task.

Returns:
    tuple[float, float]: A tuple containing:
        - micro_f1 (float): Micro-averaged F1 score (equivalent to accuracy for
            single-label multi-class classification).
        - macro_f1 (float): Macro-averaged F1 score (unweighted mean of per-class
            F1 scores).

Notes:
    - The function uses @torch.no_grad() decorator to disable gradient computation
        for efficiency during evaluation.
    - Confusion matrix is accumulated across all batches where rows represent
        true labels and columns represent predicted labels.
    - Small epsilon (1e-12) is added to denominators to prevent division by zero.
    - Only the first `batch.batch_size` predictions are used from each batch.
"""

@torch.no_grad()
def eval_f1(model, loader, device, num_classes: int):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)  # rows=true, cols=pred

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)

        pred = out.argmax(dim=-1)[: batch.batch_size]
        y    = batch.y[: batch.batch_size]

        idx = y * num_classes + pred
        cm += torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)

    tp = cm.diag().to(torch.float32)
    fp = cm.sum(dim=0).to(torch.float32) - tp
    fn = cm.sum(dim=1).to(torch.float32) - tp

    # Macro-F1
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-12)
    macro_f1 = f1_per_class.mean().item()

    # Micro-F1 (for single-label multi-class = accuracy)
    correct = tp.sum()
    total = cm.sum().to(torch.float32).clamp_min(1.0)
    micro_f1 = (correct / total).item()

    return micro_f1, macro_f1