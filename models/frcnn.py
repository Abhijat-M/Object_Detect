from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from .backbone import CustomBackbone


def build_model(num_classes):
    backbone = CustomBackbone()

    anchor_gen = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen
    )
    return model
