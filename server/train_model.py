import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from mobilenet_classifier import CatDogClassifier


def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    训练模型

    data_dir (str): 数据集根目录，应包含 train 和 val 两个子目录，每个子目录下应有 cat 和 dog 两个类别文件夹
    num_epochs (int): 训练轮数，默认为10
    batch_size (int): 批次大小，默认为32
    learning_rate (float): 学习率，默认为0.001
    """
    # 定义训练数据的预处理和数据增强流程
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # 随机裁剪并缩放到224x224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),  # 随机调整亮度、对比度和饱和度
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # 标准化
        ]
    )

    # 定义验证数据的预处理流程
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),  # 将图片缩放到256x256
            transforms.CenterCrop(224),  # 中心裁剪到224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # 标准化
        ]
    )

    # 加载训练和验证数据集
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), train_transform
    )
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 启用4个工作进程加载数据
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # 初始化模型、设备
    classifier = CatDogClassifier()
    model = classifier.model
    device = classifier.device

    # 第一阶段：冻结特征提取层，只训练分类层
    for param in model.features.parameters():
        param.requires_grad = False

    # 定义损失函数（交叉熵）和优化器（Adam）
    criterion = nn.CrossEntropyLoss()
    # 只优化未冻结的参数
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    # 训练循环
    best_acc = 0.0  # 记录最佳验证准确率

    # 第一阶段训练（仅训练分类层）
    print("第一阶段：训练分类层")
    for epoch in range(num_epochs // 2):  # 使用一半的轮次训练分类层
        print(f"Epoch {epoch+1}/{num_epochs//2}")
        print("-" * 10)

        # 训练阶段
        model.train()  # 设置为训练模式
        running_loss = 0.0
        running_corrects = 0

        # 遍历训练数据批次
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # 将数据移到指定设备（CPU/GPU）
            labels = labels.to(device)

            optimizer.zero_grad()  # 清零梯度

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计损失和正确预测数
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # 计算训练轮次的平均损失和准确率
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # 验证阶段
        model.eval()  # 设置为评估模式
        running_loss = 0.0
        running_corrects = 0

        # 在不计算梯度的情况下进行验证
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 统计损失和正确预测数
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # 计算验证集的平均损失和准确率
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 如果当前模型性能最佳，保存模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model_weights.pth")
            print("保存最佳模型")

        print()

    # 第二阶段：解冻所有层，使用较小的学习率进行微调
    print("第二阶段：微调整个模型")
    for param in model.features.parameters():
        param.requires_grad = True

    # 使用较小的学习率进行微调
    optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.1)

    for epoch in range(num_epochs // 2, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # 训练阶段
        model.train()  # 设置为训练模式
        running_loss = 0.0
        running_corrects = 0

        # 遍历训练数据批次
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # 将数据移到指定设备（CPU/GPU）
            labels = labels.to(device)

            optimizer.zero_grad()  # 清零梯度

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计损失和正确预测数
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # 计算训练轮次的平均损失和准确率
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # 验证阶段
        model.eval()  # 设置为评估模式
        running_loss = 0.0
        running_corrects = 0

        # 在不计算梯度的情况下进行验证
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 统计损失和正确预测数
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # 计算验证集的平均损失和准确率
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 如果当前模型性能最佳，保存模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model_weights.pth")
            print("保存最佳模型")

        print()

    print(f"最佳验证准确率: {best_acc:.4f}")
    return model


if __name__ == "__main__":
    data_dir = "./dataset"  # 数据集根目录，包含 train 和 val 子目录
    train_model(data_dir)  # 执行模型训练
