# 猫狗图像分类服务

这是一个基于 Flask 和 MobileNetV2 的猫狗图像分类服务。

## 环境要求

- Python 3.8+
- uv（Python 包管理器）

## 安装步骤

1. 创建并激活虚拟环境（推荐）：

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2. 安装依赖：

```bash
uv lock
uv sync
```

选择 py 解释器，地址为：`.venv/bin/python`

## 运行服务

1. 确保你在 server 目录下
2. 运行 Flask 应用：

```bash
python app.py
```

服务将在 http://localhost:5001 启动

## API 端点

- `POST /api/classify`: 分类单张图片
  - 支持文件上传或 base64 编码的图片数据
- `GET /api/health`: 健康检查
- `POST /api/batch-classify`: 批量分类图片
- `GET /api/history`: 获取分类历史记录

## 目录结构

- `app.py`: Flask 应用主文件
- `mobilenet_classifier.py`: 图像分类模型实现
- `requirements.txt`: 项目依赖
- `uploads/`: 上传文件临时存储目录
- `classified/`: 分类后的图片存储目录
  - `cat/`: 猫图片目录
  - `dog/`: 狗图片目录
