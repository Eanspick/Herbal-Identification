import io
from typing import TYPE_CHECKING, Any, cast

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.nn.functional import softmax
from typing_extensions import Buffer

from extensions import BASEDIR
from src.modules.device import DEVICE

if TYPE_CHECKING:
    from torch import Tensor  # noqa: F401

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


# Define desired height and width
desired_height = 128
desired_width = 128

RESIZE = transforms.Resize((desired_height, desired_width))
NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# Define transformation
TRANSFORM: "transforms.Compose[Tensor]" = transforms.Compose(
    [RESIZE, transforms.ToTensor(), NORMALIZE]
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
