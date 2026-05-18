import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

class Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(Classifier, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(64 * 28 * 28, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.flatten(self.block3(self.block2(self.block1(x)))))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = Classifier(num_classes=4).to(device)

model.load_state_dict(torch.load("weights/apple_disease_model.pth", map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "apple_healthy.JPG"

try:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    print(f"Анализируем фото: {image_path}")
    print(f"Диагноз: {classes[predicted_idx.item()]}")
    print(f"Уверенность нейросети: {confidence.item() * 100:.2f}%\n")
    
    print("Детальная сводка:")
    for i, cls_name in enumerate(classes):
        print(f" - {cls_name}: {probabilities[i].item() * 100:.2f}%")
        
except FileNotFoundError:
    print(f"Файл {image_path} не найден!")