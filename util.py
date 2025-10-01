# import kagglehub
# import os

# os.environ["KAGGLEHUB_CACHE"] = "./dataset"
# path = kagglehub.dataset_download("nih-chest-xrays/sample")

# print("Path to dataset files:", path)

from timm import create_model
import torch

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

print("PyTorch version:", torch.__version__)
print("CUDA version found by PyTorch:", torch.version.cuda)
print("CUDA available?:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

def predict_image(model, image_path, device, class_names, image_size=224):
    """
    Run inference on a single image.
    
    Args:
        model: trained PyTorch model
        image_path: path to the image to test
        device: 'cuda' or 'cpu'
        class_names: list of class labels
        image_size: size to resize (default=224)
    """
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225]) # ImageNet std
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]  # multilabel (sigmoid)
    
    # Top predictions
    results = [(class_names[i], float(probs[i])) for i in range(len(class_names))]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("tf_efficientnet_b4_ns", num_classes=15)
    
    # Load trained model checkpoint
    model.load_state_dict(torch.load("tf_efficientnet_b4_ns_chestxray.pth", map_location=device))
    model.to(device)

    # Example classes (you should pass your actual labels from the dataset CSV)
    class_names = ["Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema", "Cardiomegaly", "Pleural_Thickening", "Consolidation", "Pneumothorax", "Mass", "Nodule", "Atelectasis", "Effusion", "Infiltration", "No Finding"]

    image_path = r"dataset\datasets\nih-chest-xrays\sample\versions\4\sample\images\00000017_001.png"
    results = predict_image(model, image_path, device, class_names)
    result_labels = []
    result_probs = []
    print("Prediction Results:")
    for label, prob in results:
        result_labels.append(label)
        result_probs.append(prob)
        print(f"{label}: {prob:.4f}")

    fig, axis = plt.subplots(1, 2, figsize=(10, 5))
    img = Image.open(image_path).convert("RGB")
    axis[0].imshow(img)
    axis[0].axis('off')
    axis[0].set_title("Input Image")
    axis[1].barh(result_labels, result_probs)
    axis[1].set_xlabel("Probability")
    axis[1].set_ylabel("Disease")
    axis[1].set_title("Predicted Disease Probabilities")
    axis[1].set_xlim(0, 1)
    plt.tight_layout()
    plt.show()
