from classifier_model import GenderClassifier
import config
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import ClassifierDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

dataset = ClassifierDataset(root_male=config.TRAIN_DIR+"/male", root_female=config.TRAIN_DIR+"/female", transform=config.transforms)

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

classifier = GenderClassifier().to(config.DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

for epoch in range(config.NUM_EPOCHS_CLASSIFIER):
    classifier.train()
    total_loss = 0

    train_loader_tqdm = tqdm(train_loader, leave=True)

    for i, (images, labels) in enumerate(train_loader_tqdm):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS_CLASSIFIER}], Loss: {total_loss/len(train_loader):.4f}")

    classifier.eval()
    val_loss = 0.0
    val_corrects = 0
    val_samples = 0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(config.DEVICE), val_labels.to(config.DEVICE)
            val_outputs = classifier(val_images).squeeze(dim=1)
            loss = criterion(val_outputs, val_labels.float())

            val_loss += loss.item()
            val_preds = (val_outputs > 0.5).float()
            val_corrects += (val_preds == val_labels).sum().item()
            val_samples += val_labels.size(0)

    val_epoch_loss = val_loss / val_samples
    val_epoch_acc = val_corrects / val_samples
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}")

torch.save(classifier.state_dict(), "gender_classifier.pth")