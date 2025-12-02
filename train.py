import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import RetinaMNIST

# 网络引入
from models.SimpleRetinaCNN.SimpleRetinaCNN import SimpleRetinaCNN
from models.ViT_Tiny.ViT_Tiny import ViT_Tiny
from models.CNN_Transformer.CNN_Transformer import CNN_Transformer


device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 32
LR = 1e-5
EPOCHS = 20


transform_train = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

])

transform_val = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

])

# load datasets
train_dataset = RetinaMNIST(split="train", transform=transform_train, download=True,root='./data/RetinaMNIST')
val_dataset   = RetinaMNIST(split="val",   transform=transform_val, download=True,root='./data/RetinaMNIST')

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH, shuffle=False)


# model
model = CNN_Transformer(num_classes=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.squeeze().long().to(device)   # ★ 必须加 squeeze()

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Batch {x.shape[0]} | train_loss={loss.item():.4f}")

    avg_train = total_loss / len(train_loader)
    train_losses.append(avg_train)

    # ---------------- Validate ----------------
    model.eval()
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.squeeze().long().to(device)   # ★ 验证集也必须 squeeze()

            logits = model(x)
            val_total += criterion(logits, y).item()

    avg_val = val_total / len(val_loader)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")


torch.save(train_losses, "train_loss.pt")
torch.save(val_losses, "val_loss.pt")
torch.save(model.state_dict(), "model.pth")

print("训练完成（含验证集）！")
