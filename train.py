from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import winsound
from model import CurrencyCNN

def main():
    print("Script started")

    base_path = Path(r"D:\Dataset")

    if not base_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base_path}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(base_path / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(base_path / "test", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train classes: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use simple CNN instead of ResNet18
    num_classes = len(train_dataset.classes)
    model = CurrencyCNN(num_classes=num_classes)
    model = model.to(device)
    print(f"Using simple CNN with {num_classes} classes")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total * 100
        print(f"Epoch {epoch}/{num_epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%")

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100 if val_total > 0 else 0.0
        print(f"Validation accuracy: {val_acc:.2f}%")

    print("Training complete!")
    model_file = Path("model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes,
    }, model_file)
    print(f"Saved trained model to {model_file}")
    winsound.Beep(1000, 1500)  # Windows beep


if __name__ == "__main__":
    main()
