import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import random

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

class DiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.label_dict = {}
        counter = 0

        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            dirnames.sort()
            if os.path.basename(dirpath) != 'Crop_Disease_Dataset':
                subdir_name = os.path.basename(dirpath)
                self.label_dict[subdir_name] = counter
                for file in sorted(filenames):
                    self.img_labels.append(self.label_dict[subdir_name])
                    self.img_paths.append(os.path.join(dirpath, file))
                counter += 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = int(self.img_labels[idx])
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, IOError) as e:
            print(f"Skipping corrupted image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.img_paths))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def compute_mean_std(dataset, indices, image_size):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n = 0
    temp_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    for idx in indices:
        img_path = dataset.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = temp_transform(image)
            mean += image.mean(dim=[1, 2])
            std += image.std(dim=[1, 2])
            n += 1
        except Exception:
            pass
    return mean / n, std / n


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


def get_class_weights(dataset, num_classes):
    label_counts = Counter(dataset.img_labels)
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / label_counts[i] for i in range(num_classes)],
        dtype=torch.float
    )
    return weights

image_size = 64
num_classes = 13
batch_size = 32
learning_rate = 0.0001
weight_decay = 0.01
num_epochs = 60
patience = 8
input_dim = image_size * image_size * 3
root_dir = "./Crop_Disease_Dataset"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using computational hardware target: {device}")

full_dataset = DiseaseDataset(root_dir=root_dir, transform=None)
print(f"Total images found: {len(full_dataset)}")
print(f"Label dict: {full_dataset.label_dict}")

weights = get_class_weights(full_dataset, num_classes).to(device)
print(f"\nClass weights: {weights}")

all_labels = full_dataset.img_labels
all_indices = list(range(len(full_dataset)))

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_results = {
    'macro_f1': [], 'weighted_f1': [],
    'macro_precision': [], 'weighted_precision': [],
    'macro_recall': [], 'weighted_recall': [],
    'best_epochs': []
}

all_y_true, all_y_pred = [], []
all_fold_train_accuracies, all_fold_val_accuracies = [], []
all_fold_train_losses, all_fold_val_losses = [], []

label_names = [
    'Cashew Anthracnose', 'Cashew Gummosis', 'Cashew Healthy',
    'Cashew Red Rust', 'Cassava Bacterial Blight', 'Cassava Healthy',
    'Cassava Mosaic', 'Maize Healthy', 'Maize Leaf Blight',
    'Maize Streak Virus', 'Tomato Healthy', 'Tomato Septoria Leaf Spot',
    'Tomato Verticillium Wilt'
]

assert len(label_names) == num_classes, f"Label names count {len(label_names)} doesn't match num_classes {num_classes}"

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear_relu_stack(x)

model = NeuralNetwork(input_dim=input_dim, num_classes=num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Network parameters: {total_params:,}")

start = time.time()

for fold, (train_indices, val_indices) in enumerate(skf.split(all_indices, all_labels)):
    print(f"\n{'='*50}\nFOLD {fold+1}/{n_splits}\n{'='*50}")
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")

    mean, std = compute_mean_std(full_dataset, train_indices, image_size)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = DiseaseDataset(root_dir=root_dir, transform=train_transform)
    val_dataset = DiseaseDataset(root_dir=root_dir, transform=val_transform)
    train_data = torch.utils.data.Subset(train_dataset, train_indices)
    val_data = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = NeuralNetwork(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train, total_train = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # Target allocation to GPU execution

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        model.eval()
        running_val_loss = 0.0
        correct_val, total_val = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device) # Target allocation to GPU execution

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'./best_nn_fold{fold+1}.pth')
            print(f"  --> New best model saved at epoch {epoch+1}, val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  --> No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered. Best epoch: {best_epoch}")
                break

    model.load_state_dict(torch.load(f'./best_nn_fold{fold+1}.pth', map_location=device))

    model.eval()
    y_true_fold, y_pred_fold = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            y_true_fold.extend(labels.cpu().numpy())
            y_pred_fold.extend(predictions.cpu().numpy())

    all_y_true.extend(y_true_fold)
    all_y_pred.extend(y_pred_fold)

    report = classification_report(y_true_fold, y_pred_fold, target_names=label_names, digits=4, output_dict=True)

    fold_results['macro_f1'].append(report['macro avg']['f1-score'])
    fold_results['weighted_f1'].append(report['weighted avg']['f1-score'])
    fold_results['macro_precision'].append(report['macro avg']['precision'])
    fold_results['weighted_precision'].append(report['weighted avg']['precision'])
    fold_results['macro_recall'].append(report['macro avg']['recall'])
    fold_results['weighted_recall'].append(report['weighted avg']['recall'])
    fold_results['best_epochs'].append(best_epoch)

    all_fold_train_accuracies.append(train_accuracies)
    all_fold_val_accuracies.append(val_accuracies)
    all_fold_train_losses.append(train_losses)
    all_fold_val_losses.append(val_losses)

    print(f"\nFold {fold+1} Results Summary:")
    print(f"  Macro F1:    {report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}")

print(f"Total training time: {time.time() - start:.1f}s")

print(f"\n{'='*50}")
print("K-FOLD CROSS VALIDATION SUMMARY - NEURAL NETWORK")
print(f"{'='*50}")

print(f"\nMacro F1 per fold:     {[f'{x:.4f}' for x in fold_results['macro_f1']]}")
print(f"Weighted F1 per fold:  {[f'{x:.4f}' for x in fold_results['weighted_f1']]}")
print(f"Best epochs per fold:  {fold_results['best_epochs']}")

print(f"\nMacro F1:         {np.mean(fold_results['macro_f1']):.4f} +/- {np.std(fold_results['macro_f1']):.4f}")
print(f"Weighted F1:      {np.mean(fold_results['weighted_f1']):.4f} +/- {np.std(fold_results['weighted_f1']):.4f}")
print(f"Macro Precision:  {np.mean(fold_results['macro_precision']):.4f} +/- {np.std(fold_results['macro_precision']):.4f}")
print(f"Weighted Prec:    {np.mean(fold_results['weighted_precision']):.4f} +/- {np.std(fold_results['weighted_precision']):.4f}")
print(f"Macro Recall:     {np.mean(fold_results['macro_recall']):.4f} +/- {np.std(fold_results['macro_recall']):.4f}")
print(f"Weighted Recall:  {np.mean(fold_results['weighted_recall']):.4f} +/- {np.std(fold_results['weighted_recall']):.4f}")
print(f"Mean best epoch:  {np.mean(fold_results['best_epochs']):.1f} +/- {np.std(fold_results['best_epochs']):.1f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, n_splits+1), fold_results['macro_f1'], marker='o', label='Macro F1')
plt.plot(range(1, n_splits+1), fold_results['weighted_f1'], marker='s', label='Weighted F1')
plt.axhline(y=np.mean(fold_results['macro_f1']), linestyle='--', color='blue', alpha=0.5, label=f'Mean Macro F1: {np.mean(fold_results["macro_f1"]):.4f}')
plt.axhline(y=np.mean(fold_results['weighted_f1']), linestyle='--', color='orange', alpha=0.5, label=f'Mean Weighted F1: {np.mean(fold_results["weighted_f1"]):.4f}')
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('F1 Scores Across Folds - Neural Network')
plt.legend()
plt.grid(True)
plt.show()

conf_matrix = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Neural Network (All Folds Aggregated)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("Aggregated Classification Report (All Folds):")
print(classification_report(all_y_true, all_y_pred, target_names=label_names, digits=4))

folds = list(range(1, n_splits + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(folds, fold_results['macro_f1'], marker='o', label='Macro F1', color='blue')
axes[0].plot(folds, fold_results['weighted_f1'], marker='s', label='Weighted F1', color='orange')
axes[0].axhline(y=np.mean(fold_results['macro_f1']), linestyle='--', color='blue', alpha=0.4,
    label=f"Mean Macro F1: {np.mean(fold_results['macro_f1']):.4f}")
axes[0].axhline(y=np.mean(fold_results['weighted_f1']), linestyle='--', color='orange', alpha=0.4,
    label=f"Mean Weighted F1: {np.mean(fold_results['weighted_f1']):.4f}")
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('F1 Score')
axes[0].set_title('F1 Scores Across Folds - Neural Network')
axes[0].set_xticks(folds)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(folds, fold_results['macro_precision'], marker='o', label='Macro Precision', color='green')
axes[1].plot(folds, fold_results['weighted_precision'], marker='s', label='Weighted Precision', color='darkgreen')
axes[1].plot(folds, fold_results['macro_recall'], marker='^', label='Macro Recall', color='red')
axes[1].plot(folds, fold_results['weighted_recall'], marker='v', label='Weighted Recall', color='darkred')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Score')
axes[1].set_title('Precision and Recall Across Folds - Neural Network')
axes[1].set_xticks(folds)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
for ta, va in zip(all_fold_train_accuracies, all_fold_val_accuracies):
    plt.plot(ta, alpha=0.4, linestyle='--', color='blue')
    plt.plot(va, alpha=0.4, linestyle='--', color='orange')
plt.plot([], [], color='blue', label='Training Accuracy')
plt.plot([], [], color='orange', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy - Neural Network (All Folds)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
for tl, vl in zip(all_fold_train_losses, all_fold_val_losses):
    plt.plot(tl, alpha=0.4, linestyle='--', color='blue')
    plt.plot(vl, alpha=0.4, linestyle='--', color='orange')
plt.plot([], [], color='blue', label='Training Loss')
plt.plot([], [], color='orange', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - Neural Network (All Folds)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.bar(folds, fold_results['best_epochs'], color='steelblue', edgecolor='black')
plt.axhline(y=np.mean(fold_results['best_epochs']), linestyle='--', color='red', alpha=0.7,
    label=f"Mean best epoch: {np.mean(fold_results['best_epochs']):.1f}")
plt.xlabel('Fold')
plt.ylabel('Best Epoch')
plt.title('Best Epoch per Fold - Neural Network')
plt.xticks(folds)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
