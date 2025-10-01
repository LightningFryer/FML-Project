from timm import create_model
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from main import val_loader, val_ds, batch_size, num_workers
from tqdm import tqdm

y_true = []
y_pred = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model("tf_efficientnet_b4_ns", num_classes=15)
model.load_state_dict(torch.load("tf_efficientnet_b4_ns_chestxray.pth", map_location=device))
model.to(device)

model.eval()
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.sigmoid(outputs)  # multi-label probabilities
        preds = (preds > 0.5).int()     # convert to 0/1 labels

        y_true.append(labels.cpu())
        y_pred.append(preds.cpu())

y_true = torch.cat(y_true).cpu().numpy()
y_pred = torch.cat(y_pred).cpu().numpy()

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# Full breakdown per class
class_names = ["Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema", "Cardiomegaly", "Pleural_Thickening", "Consolidation", "Pneumothorax", "Mass", "Nodule", "Atelectasis", "Effusion", "Infiltration", "No Finding"]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
cmDisp = ConfusionMatrixDisplay(cm, display_labels=class_names)
cmDisp.plot()
plt.show()

y_scores = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred.ravel()
roc_auc = roc_auc_score(y_true, y_scores, average="macro")
fpr, tpr, _ = roc_curve(y_true.ravel(), y_scores.ravel())
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Macro-average")
roc_display.plot()
plt.show()