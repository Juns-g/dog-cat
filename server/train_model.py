import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from mobilenet_classifier import CatDogClassifier


def count_parameters(model):
    """
    统计模型参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, device, input_size=(3, 224, 224), num_samples=100):
    """
    测量模型推理时间

    参数:
        model: 待测试模型
        device: 运行设备
        input_size: 输入图像尺寸
        num_samples: 测试样本数量

    返回:
        average_time: 平均推理时间（毫秒）
    """
    model.eval()
    dummy_input = torch.randn(1, *input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 测量时间
    total_time = 0
    with torch.no_grad():
        for _ in range(num_samples):
            start_time = time.time()
            _ = model(dummy_input)
            total_time += (time.time() - start_time) * 1000  # 转换为毫秒

    return total_time / num_samples


def evaluate_model(model, test_loader, device):
    """
    在测试集上评估模型

    参数:
        model: 待评估模型
        test_loader: 测试数据加载器
        device: 运行设备

    返回:
        accuracy: 测试集准确率
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    训练模型

    data_dir (str): 数据集根目录，应包含 train、val 和 test 三个子目录，每个子目录下应有 cat 和 dog 两个类别文件夹
    num_epochs (int): 训练轮数，默认为10
    batch_size (int): 批次大小，默认为32
    learning_rate (float): 学习率，默认为0.001
    """
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集根目录 {data_dir} 不存在")

    # 检查必要的子目录
    required_dirs = ["train", "val"]
    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"必需的子目录 {dir_path} 不存在")

        # 检查类别子目录
        for class_name in ["cat", "dog"]:
            class_path = os.path.join(dir_path, class_name)
            if not os.path.exists(class_path):
                raise FileNotFoundError(f"类别目录 {class_path} 不存在")

            # 检查是否有有效的图片文件
            valid_extensions = (
                ".jpg",
                ".jpeg",
                ".png",
                ".ppm",
                ".bmp",
                ".pgm",
                ".tif",
                ".tiff",
                ".webp",
            )
            has_valid_files = any(
                f.lower().endswith(valid_extensions) for f in os.listdir(class_path)
            )
            if not has_valid_files:
                raise FileNotFoundError(f"在 {class_path} 中没有找到有效的图片文件")

    print("数据集检查通过，开始训练...")

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

    # 检查测试集目录
    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(test_dir):
        print("\n警告: 测试集目录不存在，将使用验证集进行最终评估")
        test_dataset = val_dataset
        test_loader = val_loader
    else:
        # 加载测试集
        test_dataset = datasets.ImageFolder(
            test_dir, val_transform  # 使用与验证集相同的变换
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

    # 加载最佳模型
    model.load_state_dict(torch.load("model_weights.pth"))

    # 评估模型性能
    test_accuracy = evaluate_model(model, test_loader, device)
    inference_time = measure_inference_time(model, device)
    param_count = count_parameters(model) / 1e6  # 转换为百万

    print("\n模型评估结果:")
    print(f"测试集准确率: {test_accuracy:.2f}%")
    print(f"单张图像推理时间: {inference_time:.2f}ms")
    print(f"模型参数量: {param_count:.2f}M")

    return model


if __name__ == "__main__":
    data_dir = "./dataset"  # 数据集根目录
    try:
        train_model(data_dir)  # 执行模型训练
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请确保数据集目录结构如下:")
        print(
            """
dataset/
├── train/
│   ├── cat/
│   │   └── (cat images)
│   └── dog/
│       └── (dog images)
├── val/
│   ├── cat/
│   │   └── (cat images)
│   └── dog/
│       └── (dog images)
└── test/  (可选)
    ├── cat/
    │   └── (cat images)
    └── dog/
        └── (dog images)
        """
        )
