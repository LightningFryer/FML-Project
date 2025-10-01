# main.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import timm
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import multiprocessing

# -------- Dataset --------
class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, disease_list, transform=None, img_col_name="Image Index"):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.disease_list = disease_list
        self.transform = transform
        self.img_col_name = img_col_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row[self.img_col_name]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Build target either from binary label columns or from a pipe-delimited Finding Labels
        if all(col in self.df.columns for col in self.disease_list):
            # dataset already has one-hot / binary columns for each disease
            labels = torch.tensor(row[self.disease_list].values, dtype=torch.float32)
        else:
            # assume a "Finding Labels" column with pipe-separated labels
            labels = torch.zeros(len(self.disease_list), dtype=torch.float32)
            if "Finding Labels" in self.df.columns:
                found = str(row["Finding Labels"]).split("|")
                for f in found:
                    if f in self.disease_list:
                        labels[self.disease_list.index(f)] = 1.0
        return image, labels

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=5, use_amp=True):
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            loader = dataloaders[phase]
            loop = tqdm(loader, desc=f"{"Training" if phase == "train" else "Validation"} epoch {epoch+1}")

            for inputs, labels in loop:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                if scaler is not None:
                    # use autocast with correct device string
                    with autocast(device.type):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(loader.dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            # step scheduler after training phase (per epoch)
            if phase == "train" and scheduler is not None:
                scheduler.step()

    return model

# -------- Utility to prepare dataframe & disease list --------
def prepare_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    # Common kaggle NIH sample CSV formats:
    # - sample_labels.csv has columns "Image Index" and "Finding Labels"
    # - Some prepared CSVs have one-hot columns per disease and an "Image" column
    if "Image Index" in df.columns and "Finding Labels" in df.columns:
        img_col = "Image Index"
        # derive disease list from all distinct labels in Finding Labels (optional), or use a fixed list
        # For reproducibility we will use a fixed list commonly used for NIH:
        disease_list = ["Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema", "Cardiomegaly", "Pleural_Thickening", "Consolidation", "Pneumothorax", "Mass", "Nodule", "Atelectasis", "Effusion", "Infiltration", "No Finding"]
    elif "Image" in df.columns:
        img_col = "Image"
        # assume rest of columns are disease label columns
        disease_list = [c for c in df.columns if c not in ("Image",)]
    else:
        # fallback: try common names
        possible_img_cols = ["Image", "image", "Image Index"]
        img_col = next((c for c in possible_img_cols if c in df.columns), df.columns[0])
        disease_list = [c for c in df.columns if c != img_col]

    return df, img_col, disease_list

# -------- Main --------
def main():
    # Paths - adjust these
    img_dir = r"dataset\datasets\nih-chest-xrays\sample\versions\4\sample\images"      # <-- change to your images folder
    csv_path = r"dataset\datasets\nih-chest-xrays\sample\versions\4\sample_labels.csv"  # <-- change to your CSV

    # Hyperparams
    batch_size = 16
    num_epochs = 20
    lr = 1e-4
    num_workers = 0  # set to 0 if you want to disable multiprocessing while debugging on Windows

    # Prepare dataframe and disease list
    df, img_col_name, disease_list = prepare_dataframe(csv_path)
    print("Image column:", img_col_name)
    print("Disease list:", disease_list)

    # If using the NIH "Finding Labels" CSV, keep that column and the image column
    if "Finding Labels" in df.columns:
        use_df = df[[img_col_name, "Finding Labels"]].copy()
    else:
        use_df = df.copy()

    train_df, val_df = train_test_split(use_df, test_size=0.2, random_state=42, shuffle=True)

    # Transforms (EfficientNet-B4 native-ish size)
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    # Datasets & loaders
    train_ds = ChestXrayDataset(train_df, img_dir, disease_list, transform=data_transforms["train"], img_col_name=img_col_name)
    val_ds = ChestXrayDataset(val_df, img_dir, disease_list, transform=data_transforms["val"], img_col_name=img_col_name)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dataloaders = {"train": train_loader, "val": val_loader}

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model (timm)
    model_name = "tf_efficientnet_b4_ns"
    model = timm.create_model(model_name, pretrained=True, num_classes=len(disease_list))
    model = model.to(device)

    # Loss, optim, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=num_epochs, use_amp=True)

    # Save
    torch.save(model.state_dict(), f"{model_name}_chestxray.pth")
    print("Training finished and model saved.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
