import torch
import matplotlib.pyplot as plt
# 画 loss 曲线
train_loss = torch.load("train_loss.pt")
val_loss = torch.load("val_loss.pt")

plt.figure(figsize=(8,5))
plt.plot(train_loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss ")
plt.grid(True)
plt.legend()
plt.savefig("loss_curve.png")

print("loss_curve.png 已保存。")
