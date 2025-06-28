import os 
import numpy as np 
import torch 
import torchvision.transforms as transforms 
from PIL import Image 
from tqdm import tqdm
import timm


def load_model(model_name):
    if model_name == "cait":
        return timm.create_model("cait_xxs24_224", pretrained=True)  
    elif model_name == "dino":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")  
    elif model_name == "deit":
        return torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    elif model_name == "tnt":
        return timm.create_model("tnt_s_patch16_224", pretrained=True)  
    elif model_name == "vit":
        return timm.create_model("vit_base_patch16_224", pretrained=True)  
    elif model_name == "swin":
        return timm.create_model("swin_base_patch4_window7_224", pretrained=True)
    else:
        raise ValueError("Modelo não suportado.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def extract_features(model, image_paths, device):
    model.to(device).eval()
    features = []
    for image_path in tqdm(image_paths, desc="Extraindo características"):
        image = transform_image(image_path).to(device)
        with torch.no_grad():
            feature = model(image)
        features.append(feature.squeeze().cpu().numpy())
    return np.array(features)

models_to_use = ["vit", "dino", "cait", "deit", "swin", "tnt"]

def create_npy_files(base_path, device):
    for model_name in models_to_use:
        model = load_model(model_name)
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):
                print(f"Processando a pasta {folder_name} com {model_name}...")
                image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png'))]
                features = extract_features(model, image_paths, device)
                labels = np.array([folder_name] * len(features))
                np.save(os.path.join(folder_path, f"{folder_name}_features_{model_name}.npy"), features)
                np.save(os.path.join(folder_path, f"{folder_name}_labels_{model_name}.npy"), labels)
                print(f"Arquivos {folder_name}_features_{model_name}.npy e {folder_name}_labels_{model_name}.npy salvos com sucesso!")

if __name__ == "__main__":
    base_path = "C:/Users/admin/Desktop/FLUO 2/base-completa"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_npy_files(base_path, device)
