import torch 
import torch.nn as nn 
from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ImageClassifier(nn.Module):
    """
    A model that will be used to classify the fashion images.
    It uses a pretrained Resnet50 model as the base, and changes the final
    linear layer to output probabilities corresponding to the number of classes
    Args:
        - num_classes (int): The number of classes in the output layer
        - model: The base model with encoder + fc output layer
        - weight (numpy array): Class weights to be used in loss function
    """
    def __init__(self, num_classes=20, model=None, weight=None):
        super(ImageClassifier, self).__init__()
        if model is not None and type(model) != str:
            self.encoder = model.encoder
            num_features = model.fc.in_features
        elif type(model) == str:
            self.encoder = getattr(models, model)()
            num_features = self.encoder.fc.in_features
        else:
            self.encoder = models.resnet50()
            num_features = self.encoder.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)
        self.encoder.fc = Identity()
        self.criterion = nn.CrossEntropyLoss(weight=None if weight is None else torch.Tensor(weight))

    def forward(self, x, y=None):
        """
        Takes the inputs (x) and labels (y) as input and returns the output
        and loss (if y is provided)
        """
        x = self.encoder(x)
        x = self.fc(x)

        if y is not None:
            loss = self.criterion(x, y)
            return x, loss
        else:
            return x
