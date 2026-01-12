import torch

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    outputs_all = []

    for images, _ in loader:
        if len(images) == 0:
            continue
        images = [i.to(device) for i in images]
        outputs = model(images)
        outputs_all.extend(
            [{k:v.cpu() for k,v in o.items()} for o in outputs]
        )

    return outputs_all
