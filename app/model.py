import io
from typing import Any, cast

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor, nn
from torch.nn.functional import softmax
from typing_extensions import Buffer

from extensions import BASEDIR
from src.modules.dataset import NORMALIZE, RESIZE
from src.modules.device import DEVICE
from src.modules.preprocess import SubtractBackground

MODEL_PATH = BASEDIR / "models"


class HerbalIdentificationModel(nn.Module):
    def __init__(self):
        super().__init__()  # type: ignore
        self.network = models.resnet34(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb: Any):
        out = self.network(xb)
        return out


TRANSFORM: "transforms.Compose[Tensor]" = transforms.Compose(
    [
        SubtractBackground(),
        RESIZE,
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


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
    tensor = TRANSFORM(img_pil)
    xb = tensor.unsqueeze(0)
    yb = model(xb)
    probs = softmax(yb, dim=1)
    top_prob, top_class = torch.max(probs, dim=1)
    confidence = top_prob.item()
    index = cast(int, top_class[0].item())
    prediction = CLASSES[index] if index < len(CLASSES) else "Other"
    print("\n", prediction, confidence, "\n")

    if confidence < threshold or index >= len(CLASSES):
        raise ValueError("Uploaded image is not recognized as a herbal image.")

    return prediction, confidence
