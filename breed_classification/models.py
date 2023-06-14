import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# Create custom stack ensemble model
class EnsembleModel(nn.Module):
  def __init__(self):
    super().__init__()
    # defined pretrained base model
    self.VGG19 = models.vgg19(pretrained=True)
    self.RES101 = models.resnet101(pretrained=True)

    # Get input features of last layer of base models
    vgg19_num_ftrs = self.VGG19.classifier[6].in_features
    res101_num_ftrs = self.RES101.fc.in_features

    # Change last layer to have the same number of output before forward to 
    self.VGG19.classifier[6] = nn.Linear(vgg19_num_ftrs, 1024)
    self.RES101.fc = nn.Linear(res101_num_ftrs, 1024)

    # Define classification output
    self.relu=nn.ReLU(inplace=True)
    self.classifier = nn.Linear(2048, 133)

  def forward(self, x_inp):
    # Get output from base models
    x1 = self.VGG19(x_inp)
    x2 = self.RES101(x_inp)
    # Ensemble information output of base models, could be concat, avg
    x = torch.cat((x1, x2), dim=1)
    x = self.classifier(self.relu(x))
    return x