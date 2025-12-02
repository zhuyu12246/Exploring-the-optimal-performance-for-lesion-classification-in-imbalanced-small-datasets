import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import RetinaMNIST

from models.SimpleRetinaCNN.SimpleRetinaCNN import SimpleRetinaCNN
from models.ViT_Tiny.ViT_Tiny import ViT_Tiny

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# --------------------------
# Dataset / DataLoader
# --------------------------
test_dataset = RetinaMNIST(
    split="test",
    transform=transform,
    download=True,
    root='./data/RetinaMNIST'
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_classes = int(test_dataset.labels.max() + 1)

# --------------------------
# Load Model
# --------------------------
model = ViT_Tiny(num_classes = 5).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# --------------------------
# Test Phase
# --------------------------
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:

        x = x.to(device)
        y = y.long().to(device).view(-1)   # 保证标签是 (batch,) 一维向量

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        # 收集混淆矩阵数据
        all_preds.extend(pred.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

        correct += (pred == y).sum().item()
        total += y.size(0)

# 准确率
acc = correct / total
print(f"\nTest Accuracy = {acc * 100:.2f}%")
print("len(all_labels):", len(all_labels))
print("len(all_preds):", len(all_preds))
print("total samples:", len(test_dataset))

# --------------------------
# Confusion Matrix
# --------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)

# 在每个格子里写数字
thresh = cm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(cm[i, j], 'd'),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

