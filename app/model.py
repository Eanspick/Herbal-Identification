import io
from typing import Any

import torch
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.nn.functional import softmax
from typing_extensions import Buffer

from extensions import BASEDIR
from src.modules.dataset import TRANSFORM, UNIQUE_PLANTS
from src.modules.device import DEVICE

MODEL_PATH = BASEDIR / "models"


class TransformError(Exception): ...


class HerbalIdentificationModel(nn.Module):
    def __init__(self):
        super().__init__()  # type: ignore
        self.network = models.resnet34(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb: Any):
        out = self.network(xb)
        return out


CLASSES = [
    "Augmented Arjun Leaf",
    "Augmented Curry Leaf",
    "Augmented Marsh Pennywort Leaf",
    "Augmented Mint Leaf",
    "Augmented Neem Leaf",
    "Augmented Rubble Leaf",
]


model = HerbalIdentificationModel()
model.load_state_dict(
    torch.load(  # type: ignore
        MODEL_PATH / "herbal-identification-new-resnet34.pth",
        map_location=DEVICE,
    )
)
model.eval()


def predict_image(img: Buffer, threshold: float = 0.6):
    img_pil = Image.open(io.BytesIO(img))

    try:
        tensor = TRANSFORM(img_pil)
    except RuntimeError as e:
        raise TransformError(e) from e

    xb = tensor.unsqueeze(0)
    yb = model(xb)
    probs = softmax(yb, dim=1)
    top_prob, top_class = torch.max(probs, dim=1)
    confidence = top_prob.item()
    index = int(top_class[0].item())
    prediction = CLASSES[index] if index < len(CLASSES) else UNIQUE_PLANTS[index]
    print("\n", prediction, confidence, "\n")

    # if confidence < threshold or index >= len(CLASSES):+
    if index >= len(CLASSES):
        raise ValueError("Uploaded image is not recognized as a herbal image.")

    return prediction, confidence
