import torch
from torchreid.models import build_model # type: ignore
from torchvision import transforms
from PIL import Image

class OSNetReID:

    def __init__(self, device="cuda"):
        self.device = device
        self.model = build_model(
            name='osnet_x1_0',
            num_classes=1000,  # not used for embeddings
            pretrained=True,
        )
        self.model.to(device)
        self.model.eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, frame, box):
        """
        Crop a person from the frame using box and return normalized embedding.
        Args:
            frame: PIL Image
            box: [x1, y1, x2, y2]
        Returns:
            embedding: torch tensor, shape [1, feature_dim]
        """
        x1, y1, x2, y2 = map(int, box)
        person_crop = frame.crop((x1, y1, x2, y2))
        tensor = self.transform(person_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb
