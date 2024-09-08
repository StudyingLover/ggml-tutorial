import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# 检查是否有可用的 CUDA 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义一个简单的神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 数据预处理和加载
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 创建模型、损失函数和优化器，并将它们移动到 GPU 上
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

print("训练完成")

# 保存模型
torch.save(model.state_dict(), "model/mnist_model.pth")

# 推理output_images/random_image.png
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 读取图片
image = Image.open("example.png").convert("L")
image = transform(image).unsqueeze(0).to(device)

# 加载模型
model = CNN().to(device)
model.load_state_dict(torch.load("model/mnist_model.pth"))
model.eval()

# 进行推理
output = model(image)
_, predicted = torch.max(output, 1)

# 显示图片和预测结果
plt.imshow(np.array(image.cpu().numpy().reshape(28, 28)), cmap="gray")
plt.title(f"Predicted: {predicted.item()}")

plt.show()
