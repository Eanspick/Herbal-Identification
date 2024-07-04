import torchvision.models as models
from torchinfo import summary

# network = nn.Sequential(
#     nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),  # output : 64*64*64
#     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),  # output : 128*32*32
#     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),  # output : 256*16*16
#     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),  # output : 512*8*8
#     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),  # output : 1024*4*4
#     nn.AdaptiveAvgPool2d(1),
#     nn.Flatten(),
#     nn.Linear(1024, 512),
#     nn.ReLU(),
#     nn.Linear(512, 256),
#     nn.ReLU(),
#     nn.Linear(256, 38),
# )

summary(models.resnet34())
