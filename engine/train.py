import torch


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    steps = 0

    for images, targets in loader:
        if len(images) == 0:
            continue

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)
