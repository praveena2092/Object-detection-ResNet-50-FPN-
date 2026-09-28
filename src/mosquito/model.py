import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes_with_bg: int, pretrained: bool = True):
    """Faster R-CNN, ResNet-50 + FPN; box head replaced for our classes (+ background)."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    return model


def load_checkpoint(path: str, num_classes_with_bg: int, device):
    model = build_model(num_classes_with_bg, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
