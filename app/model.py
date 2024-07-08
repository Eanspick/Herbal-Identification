import io
from typing import Any

import torch
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.nn.functional import softmax
from typing_extensions import Buffer

from extensions import BASEDIR
from src.modules.dataset import TRANSFORM
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


PLANT_DICT = {
    "Arjun Leaf": "inc/arjun_leaf.inc.html",
    "Curry Leaf": "inc/curry_leaf.inc.html",
    "Marsh Pennywort Leaf": "inc/marsh_pennywort_leaf.inc.html",
    "Mint Leaf": "inc/mint_leaf.inc.html",
    "Neem Leaf": "inc/neem_leaf.inc.html",
    "Rubble Leaf": "inc/rubble_leaf.inc.html",
    "Flavia Plant": "inc/z.inc.html",
    "Random Image": "inc/zz.inc.html",
}

CLASSES = list(PLANT_DICT)


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
    prediction = CLASSES[index]
    print("\n", prediction, confidence, "\n")

    # if confidence < threshold or index >= len(CLASSES):+
    # if index >= len(CLASSES):
    #     raise ValueError("Uploaded image is not recognized as a herbal image.")

    return prediction, confidence
