# 导入所需的库
import torch  # PyTorch 核心库，用于张量计算和神经网络
import torch.nn as nn  # PyTorch 神经网络模块，包含构建网络的层和函数
import torchvision.models as models  # 包含预训练模型的库，如 MobileNetV2
import torchvision.transforms as transforms  # 提供常用的图像预处理操作
from PIL import Image  # Python Imaging Library (Pillow)，用于图像文件的打开和处理
import os  # 提供与操作系统交互的功能，如文件路径操作、目录创建等
import shutil  # 提供高级文件操作，如复制文件
import ssl  # 处理 SSL/TLS 加密通信，这里用于解决模型下载时的证书验证问题

# 临时禁用 SSL 验证（仅用于开发环境）
# 在某些环境下，下载预训练模型时可能会遇到 SSL 证书验证错误
# 这行代码创建了一个不验证证书的 SSL 上下文，绕过这个问题
# 注意：这在生产环境中可能存在安全风险
ssl._create_default_https_context = ssl._create_unverified_context


class CatDogClassifier:
    def __init__(self):
        # 加载预训练的 MobileNetV2 模型
        self.model = models.mobilenet_v2(
            # 指定使用在 ImageNet 数据集上预训练的权重
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # 修改最后的全连接层 (classifier)
        # MobileNetV2 原始输出是 ImageNet 的 1000 个类别，将其改为 2 个类别（猫和狗）
        # self.model.last_channel 获取模型最后一个卷积层输出的通道数
        # nn.Linear 创建一个线性层（全连接层），输入维度是 last_channel，输出维度是 2
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)

        # 检查是否有可用的 GPU (CUDA)，如果没有则使用 CPU
        # torch.cuda.is_available() 检查 CUDA 是否可用
        # torch.device() 创建一个设备对象，指定模型和数据应在哪个设备上运行
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 将模型移动到指定的设备 (GPU 或 CPU)
        self.model.to(self.device)

        # 尝试加载本地训练好的模型权重文件 (model_weights.pth)，如果成功加载，会覆盖预训练的权重，使用我们自己训练的结果
        # map_location=self.device 确保权重加载到正确的设备上
        try:
            self.model.load_state_dict(
                torch.load("model_weights.pth", map_location=self.device)
            )
            print("本地模型权重文件 model_weights.pth 加载成功。")
        except:
            print(
                "本地模型权重文件 model_weights.pth 加载失败，使用 ImageNet 预训练的权重。"
            )

        # 设置模型为评估模式 (evaluation mode)
        # 这会关闭 Dropout 和 Batch Normalization 的训练特定行为
        # 对于推理（预测）是必要的步骤
        self.model.eval()

        # 定义图像预处理流程
        # transforms.Compose 将多个预处理步骤组合在一起
        self.transform = transforms.Compose(
            [
                #  将图像短边调整到 256 像素，长边按比例缩放
                transforms.Resize(256),
                # 从图像中心裁剪出 224x224 大小的区域
                # 这是 MobileNetV2 通常要求的输入尺寸
                transforms.CenterCrop(224),
                # 将 PIL 图像或 NumPy 数组转换为 PyTorch 张量
                # 并将像素值从 [0, 255] 缩放到 [0.0, 1.0]
                transforms.ToTensor(),
                # 使用 ImageNet 数据集的均值和标准差对图像进行标准化
                # 这是预训练模型要求的标准步骤，有助于模型更好地工作
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 定义分类标签列表，索引 0 对应 'cat'，索引 1 对应 'dog'
        self.classes = ["cat", "dog"]

    # 定义预测单张图片类别的方法
    def predict_image(self, image_path):
        """
        预测单张图片的类别

        参数:
            image_path: 图片的文件路径 (字符串) 或已打开的 PIL 图像对象

        返回:
            字典，包含预测类别 ('class') 和置信度 ('confidence')，或错误信息 ('error')
        """
        try:
            # 检查输入是文件路径还是 PIL 图像对象
            if isinstance(image_path, str):
                # 如果是路径，使用 PIL 打开图像，并转换为 RGB 格式
                image = Image.open(image_path).convert("RGB")
            else:
                # 如果是 PIL 对象，直接使用并确保是 RGB 格式
                image = image_path.convert("RGB")

            # 应用之前定义的预处理流程，并将图像转换为张量
            # .unsqueeze(0) 在最前面增加一个维度，将 [C, H, W] 变为 [1, C, H, W]
            # 这是因为模型通常期望接收一个批次 (batch) 的图像，即使只有一张也要构造成批次
            # .to(self.device) 将张量移动到模型所在的设备 (GPU 或 CPU)
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 使用 torch.no_grad() 上下文管理器
            # 在这个块内部，PyTorch 不会计算梯度，可以节省内存并加速计算
            # 这在推理（预测）时是标准的做法
            with torch.no_grad():
                # 将处理好的图像张量输入模型，得到输出 (logits)
                outputs = self.model(image_tensor)
                # torch.max(outputs, 1) 找到输出张量中第二个维度（类别维度）上的最大值的索引
                # _ 接收最大值本身（我们不需要），predicted 接收最大值对应的索引 (0 或 1)
                _, predicted = torch.max(outputs, 1)
                # torch.nn.functional.softmax(outputs, dim=1) 将模型的原始输出 (logits) 转换为概率分布
                # dim=1 表示在类别维度上计算 softmax
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # 获取预测类别的置信度（概率）
                # probabilities[0] 获取批次中第一张（也是唯一一张）图像的概率
                # [predicted.item()] 使用预测的类别索引 (0 或 1) 来获取对应的概率值
                # .item() 将只有一个元素的张量转换为 Python 数值
                confidence = probabilities[0][predicted.item()].item()

            # 返回包含预测类别和置信度的字典
            # self.classes[predicted.item()] 使用预测索引从标签列表中获取类别名称 ('cat' 或 'dog')
            # round(confidence * 100, 2) 将置信度转换为百分比并保留两位小数
            return {
                "class": self.classes[predicted.item()],
                "confidence": round(confidence * 100, 2),
            }
        # 如果在处理过程中发生任何异常，捕获它并返回包含错误信息的字典
        except Exception as e:
            return {"error": str(e)}

    # 定义批量分类图片的方法
    def batch_classify(self, input_dir, output_cat_dir, output_dog_dir):
        """
        批量对文件夹中的图片进行分类，并将图片复制到对应的猫/狗文件夹

        参数:
            input_dir: 包含待分类图片的输入文件夹路径
            output_cat_dir: 分类为猫的图片的输出文件夹路径
            output_dog_dir: 分类为狗的图片的输出文件夹路径

        返回:
            包含分类结果统计信息的字典
        """
        # 检查输出文件夹是否存在，如果不存在则创建
        if not os.path.exists(output_cat_dir):
            os.makedirs(output_cat_dir)
        if not os.path.exists(output_dog_dir):
            os.makedirs(output_dog_dir)

        # 初始化用于存储统计结果的字典
        results = {
            "total": 0,  # 处理的总图片数
            "cat_count": 0,  # 分类为猫的数量
            "dog_count": 0,  # 分类为狗的数量
            "errors": 0,  # 处理出错的数量
            "details": [],  # 存储每张图片分类详情的列表
        }

        # 遍历输入文件夹中的所有文件和子目录
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                try:
                    # 构建完整的图片文件路径
                    image_path = os.path.join(input_dir, filename)
                    # 调用 predict_image 方法对当前图片进行预测
                    prediction = self.predict_image(image_path)

                    if "error" in prediction:
                        results["errors"] += 1
                        print(f"Error processing {filename}: {prediction['error']}")
                        continue

                    results["total"] += 1
                    # 获取预测的类别和置信度
                    predicted_class = prediction["class"]
                    confidence = prediction["confidence"]

                    results["details"].append(
                        {
                            "filename": filename,
                            "class": predicted_class,
                            "confidence": confidence,
                        }
                    )

                    # 根据预测的类别，将图片复制到对应的输出文件夹
                    if predicted_class == "cat":
                        results["cat_count"] += 1
                        shutil.copy(image_path, os.path.join(output_cat_dir, filename))
                    else:
                        results["dog_count"] += 1
                        shutil.copy(image_path, os.path.join(output_dog_dir, filename))

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    results["errors"] += 1

        # 返回包含所有统计结果的字典
        return results


# 其他模块可以直接使用 `from mobilenet_classifier import classifier` 来获取这个实例并调用其方法
classifier = CatDogClassifier()
