def count_params(model) -> int:
    # DDP syncs gradients for trainable params only
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params
