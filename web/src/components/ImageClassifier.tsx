/**
 * 单张图片分类组件
 * 允许用户上传或拖拽图片进行分类
 * 增强了错误处理和用户反馈
 */
import React, { useState, useCallback } from "react";
import {
  Card,
  Button,
  message,
  Spin,
  Typography,
  Progress,
  Divider,
  Alert,
  Space,
} from "antd";
import { InboxOutlined, ReloadOutlined } from "@ant-design/icons";
import { useDropzone } from "react-dropzone";
import { classifyImage } from "../api";
import { ClassificationResult } from "../api/types";
import { colorMap, textMap } from "@/constant";

const { Title, Text, Paragraph } = Typography;

const ImageClassifier: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [classificationResult, setClassificationResult] =
    useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);

  // 处理文件上传
  const handleFileUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setErrorDetails(null);
    setClassificationResult(null);

    try {
      // 读取文件为 Base64
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = async () => {
        const base64Image = reader.result as string;
        setImageUrl(base64Image);

        try {
          // 使用 API 函数进行分类
          const result = await classifyImage(base64Image);
          setClassificationResult(result);
        } catch (err) {
          console.error("Classification error:", err);
          setError("图像分类失败，请重试。");
        } finally {
          setLoading(false);
        }
      };

      reader.onerror = () => {
        setError("读取文件失败，请重试。");
        setLoading(false);
      };
    } catch (err) {
      console.error("File handling error:", err);
      setError("处理文件时出错，请重试。");
      setLoading(false);
    }
  };

  // 使用 react-dropzone 处理拖放
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.type.startsWith("image/")) {
        handleFileUpload(file);
      } else {
        message.error("请上传图片文件！");
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png", ".gif"],
    },
    maxFiles: 1,
  });

  // 清除当前结果
  const handleClear = () => {
    setImageUrl(null);
    setClassificationResult(null);
    setError(null);
    setErrorDetails(null);
  };

  // 重试上传分类
  const handleRetry = () => {
    if (imageUrl) {
      // 如果有图片但分类失败，只重试分类请求
      setLoading(true);
      setError(null);
      setErrorDetails(null);

      // 从imageUrl获取base64数据
      classifyImage(imageUrl)
        .then((response) => {
          setClassificationResult(response);
          setLoading(false);
        })
        .catch((err) => {
          console.error("Retry classification error:", err);
          setError("图像分类重试失败。");
          setLoading(false);
        });
    }
  };

  return (
    <div className="image-classifier">
      <Card title="上传图片进行分类" bordered={false}>
        {!imageUrl && (
          <div {...getRootProps()} className="dropzone-area">
            <input {...getInputProps()} />
            <p>
              <InboxOutlined className="dropzone-icon" />
            </p>
            <p className="dropzone-text">
              {isDragActive ? "放下图片以上传" : "点击或拖拽图片到此区域上传"}
            </p>
            <p className="dropzone-hint">支持 JPG、PNG、GIF 格式</p>
          </div>
        )}

        {imageUrl && (
          <div className="result-area">
            <div className="image-container">
              <img src={imageUrl} alt="上传图片" className="preview-image" />
            </div>

            <Divider />

            {loading && (
              <div className="loading-container">
                <Spin size="large" />
                <Text className="loading-text">正在分析图片...</Text>
              </div>
            )}

            {error && (
              <div className="error-container">
                <Alert
                  message="分类错误"
                  description={
                    <div>
                      <p>{error}</p>
                      {errorDetails && (
                        <details>
                          <summary>详细错误信息</summary>
                          <pre className="error-details">{errorDetails}</pre>
                        </details>
                      )}
                    </div>
                  }
                  type="error"
                  showIcon
                  action={
                    <Button
                      icon={<ReloadOutlined />}
                      size="small"
                      onClick={handleRetry}
                    >
                      重试
                    </Button>
                  }
                />
              </div>
            )}

            {classificationResult && !loading && (
              <div className="classification-result">
                <Title level={3} className="result-title">
                  分类结果:
                </Title>
                <div className="result-details">
                  <Title level={4} className="classification-label">
                    {textMap[classificationResult.class]}
                  </Title>
                  <Paragraph className="confidence-text">
                    置信度: {classificationResult.confidence}%
                  </Paragraph>
                  <Progress
                    percent={classificationResult.confidence}
                    status="active"
                    strokeColor={colorMap[classificationResult.class]}
                  />
                </div>
              </div>
            )}

            <div className="action-buttons">
              <Space>
                <Button type="primary" onClick={handleClear}>
                  清除并上传新图片
                </Button>
                {error && (
                  <Button
                    type="primary"
                    icon={<ReloadOutlined />}
                    onClick={handleRetry}
                  >
                    重试当前图片
                  </Button>
                )}
              </Space>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

export default ImageClassifier;
