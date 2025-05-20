# ======= predict.py =======
import torch
from torchvision import transforms
from PIL import Image
import pickle
from model import get_cnn_model

def predict(image_path):
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    model = get_cnn_model(num_classes=len(le.classes_))
    model.load_state_dict(torch.load("ocr_cnn_model.pth", map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return le.inverse_transform([predicted.item()])[0]