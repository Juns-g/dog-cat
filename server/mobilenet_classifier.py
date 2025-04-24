import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil
import ssl

# 临时禁用 SSL 验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context


class CatDogClassifier:
    def __init__(self):
        # 加载预训练的 MobileNetV2 模型
        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # 修改最后的全连接层，将输出改为2个类别（猫和狗）
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)

        # 检查是否有可用的 GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 尝试加载训练好的模型权重，如果没有则使用预训练模型
        try:
            self.model.load_state_dict(
                torch.load("model_weights.pth", map_location=self.device)
            )
            print("Loaded saved model weights.")
        except:
            print(
                "Using pretrained model. Note: For better accuracy, train the model specifically on cat/dog dataset."
            )

        # 设置模型为评估模式
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 分类标签
        self.classes = ["cat", "dog"]

    def predict_image(self, image_path):
        """
        预测单张图片的类别

        参数:
            image_path: 图片的路径或已打开的PIL图像对象

        返回:
            字典，包含预测类别和置信度
        """
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path.convert("RGB")

            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted.item()].item()

            return {
                "class": self.classes[predicted.item()],
                "confidence": round(confidence * 100, 2),
            }
        except Exception as e:
            return {"error": str(e)}

    def batch_classify(self, input_dir, output_cat_dir, output_dog_dir):
        """
        批量对文件夹中的图片进行分类，并移动到对应的猫/狗文件夹

        参数:
            input_dir: 输入图片文件夹
            output_cat_dir: 猫图片的输出文件夹
            output_dog_dir: 狗图片的输出文件夹

        返回:
            分类结果统计
        """
        if not os.path.exists(output_cat_dir):
            os.makedirs(output_cat_dir)
        if not os.path.exists(output_dog_dir):
            os.makedirs(output_dog_dir)

        results = {
            "total": 0,
            "cat_count": 0,
            "dog_count": 0,
            "errors": 0,
            "details": [],
        }

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                try:
                    image_path = os.path.join(input_dir, filename)
                    prediction = self.predict_image(image_path)

                    if "error" in prediction:
                        results["errors"] += 1
                        continue

                    results["total"] += 1
                    predicted_class = prediction["class"]
                    confidence = prediction["confidence"]

                    # 记录分类详情
                    results["details"].append(
                        {
                            "filename": filename,
                            "class": predicted_class,
                            "confidence": confidence,
                        }
                    )

                    # 复制文件到对应文件夹
                    if predicted_class == "cat":
                        results["cat"] += 1
                        shutil.copy(image_path, os.path.join(output_cat_dir, filename))
                    else:
                        results["dog"] += 1
                        shutil.copy(image_path, os.path.join(output_dog_dir, filename))

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    results["errors"] += 1

        return results


# 创建分类器实例 - 在导入模块时自动初始化
classifier = CatDogClassifier()
