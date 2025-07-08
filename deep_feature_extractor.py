import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device

        # Load a pre-trained ResNet-18 and remove final classifier
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(backbone.children())[:-1]  # remove final FC layer
        self.model = nn.Sequential(*modules)
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),  # smaller for speed
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img_crop):
        """
        img_crop: numpy array (H, W, 3) in BGR format (OpenCV)
        Returns: feature vector as numpy array (512,)
        """
        # Convert BGR to RGB
        img_rgb = img_crop[..., ::-1]
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor).squeeze()

        return feat.cpu().numpy()