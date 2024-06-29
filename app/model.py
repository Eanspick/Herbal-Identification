import io
from typing import cast

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.nn.functional import softmax
from typing_extensions import Buffer

from extensions import BASEDIR

MODEL_PATH = BASEDIR / "models"


class HerbalIdentificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        out = self.network(xb)
        return out


transform = transforms.Compose([transforms.Resize(size=128), transforms.ToTensor()])

num_classes = [
    "Augmented Arjun Leaf",
    "Augmented Curry Leaf",
    "Augmented Marsh Pennywort Leaf",
    "Augmented Mint Leaf",
    "Augmented Neem Leaf",
    "Augmented Rubble Leaf",
]


model = HerbalIdentificationModel()
model.load_state_dict(
    torch.load(
        MODEL_PATH / "herbal-identification-resnet34.pth",
        map_location=torch.device("cpu"),
    )
)
model.eval()


def predict_image(img: Buffer, threshold: float = 0.6):
    img_pil = Image.open(io.BytesIO(img))
    tensor = transform(img_pil)
    xb = tensor.unsqueeze(0)
    yb = model(xb)
    probs = softmax(yb, dim=1)
    top_prob, top_class = torch.max(probs, dim=1)
    confidence = top_prob.item()
    prediction = num_classes[cast(int, top_class[0].item())]
    print("\n\n\n\n\n\n\n")
    print(prediction, confidence)
    print("\n\n\n\n\n\n\n")

    if confidence < threshold:
        raise ValueError("Uploaded image is not recognized as a herbal image.")

    return prediction, confidence
