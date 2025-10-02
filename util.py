from timm import create_model
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

print("PyTorch version:", torch.__version__)
print("CUDA version found by PyTorch:", torch.version.cuda)
print("CUDA available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# ---------------- Grad-CAM class ---------------- #
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # forward hook
        target_layer.register_forward_hook(self.save_activation)
        # backward hook
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # resize to input size
        cam = torch.nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ---------------- Prediction + Grad-CAM ---------------- #
def predict_and_explain(model, image_path, device, class_names, image_size=224):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # preprocess
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # predictions
    results = [(class_names[i], float(probs[i])) for i in range(len(class_names))]
    results.sort(key=lambda x: x[1], reverse=True)

    # ----- Grad-CAM -----
    # Hook into last conv layer of EfficientNet-B4
    target_layer = model.conv_head
    gradcam = GradCAM(model, target_layer)

    # Run Grad-CAM for top prediction
    top_class_idx = np.argmax(probs)
    cam = gradcam.generate(input_tensor, class_idx=top_class_idx)

    # Convert original image to numpy
    img_np = np.array(image.resize((image_size, image_size))) / 255.0

    # Heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * heatmap.astype(np.float32) / 255 + img_np).clip(0, 1)

    # ---- Plot ----
    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    axis[0].imshow(img_np)
    axis[0].axis("off")
    axis[0].set_title("Input Image")

    axis[1].barh([r[0] for r in results], [r[1] for r in results])
    axis[1].set_xlim(0, 1)
    axis[1].set_title("Predicted Probabilities")

    axis[2].imshow(overlay)
    axis[2].axis("off")
    axis[2].set_title(f"Grad-CAM: {class_names[top_class_idx]}")

    plt.tight_layout()
    plt.show()

    return results

# ---------------- Main ---------------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("tf_efficientnet_b4_ns", num_classes=15)
    model.load_state_dict(torch.load("tf_efficientnet_b4_ns_chestxray_10_epochs.pth", map_location=device))
    model.to(device)

    class_names = ["Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema", 
                   "Cardiomegaly", "Pleural_Thickening", "Consolidation", 
                   "Pneumothorax", "Mass", "Nodule", "Atelectasis", 
                   "Effusion", "Infiltration", "No Finding"]

    image_path = "effusion.png"
    results = predict_and_explain(model, image_path, device, class_names)
    print("Prediction Results:")
    for label, prob in results:
        print(f"{label}: {prob:.4f}")
