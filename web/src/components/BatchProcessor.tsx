/**
 * 批量图片处理组件
 * 允许用户指定文件夹进行批量分类
 * 增强了错误处理和用户反馈
 */
import React, { useState } from "react";
import {
  Card,
  Form,
  Input,
  Button,
  Alert,
  Table,
  Typography,
  Spin,
  Divider,
  Space,
} from "antd";
import { FolderOpenOutlined, ReloadOutlined } from "@ant-design/icons";
import axios from "axios";
import { BatchClassificationResult } from "../../../shared/types";

const { Title, Text, Paragraph } = Typography;

const BatchProcessor: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BatchClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);
  const [lastSubmitValues, setLastSubmitValues] = useState<any>(null);

  // 提交批量处理请求
  const handleSubmit = async (values: {
    inputDir: string;
    outputCatDir?: string;
    outputDogDir?: string;
  }) => {
    setLoading(true);
    setError(null);
    setErrorDetails(null);
    setResult(null);
    setLastSubmitValues(values);

    try {
      const response = await axios.post(
        `${process.env.AIPA_API_DOMAIN}/api/batch-classify`,
        {
          input_dir: values.inputDir,
          output_cat_dir: values.outputCatDir || "",
          output_dog_dir: values.outputDogDir || "",
        }
      );

      setResult(response.data);
    } catch (err: any) {
      console.error("Batch processing error:", err);

      let errorMessage = "批量处理失败，请检查文件夹路径是否正确并重试。";
      let errorDetail = null;

      // 从错误响应中提取详细信息
      if (err.response) {
        // 服务器响应了错误状态码
        const { data, status } = err.response;

        if (data && data.message) {
          errorMessage = data.message;
        }

        if (data && data.details) {
          errorDetail = JSON.stringify(data.details, null, 2);
        }
      } else if (err.request) {
        // 请求已发送但没有收到响应
        errorMessage = "无法连接到服务器，请检查网络连接或服务器状态。";
      } else {
        // 其他错误
        errorMessage = `请求设置错误: ${err.message}`;
      }

      setError(errorMessage);
      if (errorDetail) {
        setErrorDetails(errorDetail);
      }
    } finally {
      setLoading(false);
    }
  };

  // 重试上一次请求
  const handleRetry = () => {
    if (lastSubmitValues) {
      handleSubmit(lastSubmitValues);
    }
  };

  // 结果表格列定义
  const columns = [
    {
      title: "文件名",
      dataIndex: "filename",
      key: "filename",
    },
    {
      title: "分类结果",
      dataIndex: "classification",
      key: "classification",
      render: (text: string) => (
        <span className={`classification-tag ${text}`}>
          {text === "cat" ? "🐱 猫" : "🐶 狗"}
        </span>
      ),
    },
    {
      title: "置信度",
      dataIndex: "confidence",
      key: "confidence",
      render: (confidence: number) => `${confidence}%`,
    },
  ];

  return (
    <div className="batch-processor">
      <Card title="批量处理图片" bordered={false}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            inputDir: "",
            outputCatDir: "",
            outputDogDir: "",
          }}
        >
          <Form.Item
            name="inputDir"
            label="输入文件夹路径"
            rules={[{ required: true, message: "请输入图片文件夹路径" }]}
            help="包含要分类的图片的文件夹路径"
          >
            <Input
              prefix={
                <FolderOpenOutlined
                  onPointerEnterCapture={undefined}
                  onPointerLeaveCapture={undefined}
                />
              }
              placeholder="例如: C:\Images\ToClassify"
            />
          </Form.Item>

          <Form.Item
            name="outputCatDir"
            label="猫图片输出文件夹（可选）"
            help="分类为猫的图片将被复制到此文件夹，留空将使用默认文件夹"
          >
            <Input
              prefix={
                <FolderOpenOutlined
                  onPointerEnterCapture={undefined}
                  onPointerLeaveCapture={undefined}
                />
              }
              placeholder="例如: C:\Images\Cats（留空则使用默认文件夹）"
            />
          </Form.Item>

          <Form.Item
            name="outputDogDir"
            label="狗图片输出文件夹（可选）"
            help="分类为狗的图片将被复制到此文件夹，留空将使用默认文件夹"
          >
            <Input
              prefix={
                <FolderOpenOutlined
                  onPointerEnterCapture={undefined}
                  onPointerLeaveCapture={undefined}
                />
              }
              placeholder="例如: C:\Images\Dogs（留空则使用默认文件夹）"
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                开始批量处理
              </Button>
              {error && lastSubmitValues && (
                <Button
                  icon={
                    <ReloadOutlined
                      onPointerEnterCapture={undefined}
                      onPointerLeaveCapture={undefined}
                    />
                  }
                  onClick={handleRetry}
                  disabled={loading}
                >
                  重试上次操作
                </Button>
              )}
            </Space>
          </Form.Item>
        </Form>

        {loading && (
          <div className="loading-container">
            <Spin size="large" />
            <Text className="loading-text">正在处理文件夹中的图片...</Text>
          </div>
        )}

        {error && (
          <div className="error-container">
            <Alert
              message="批量处理错误"
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
                  icon={
                    <ReloadOutlined
                      onPointerEnterCapture={undefined}
                      onPointerLeaveCapture={undefined}
                    />
                  }
                  size="small"
                  onClick={handleRetry}
                >
                  重试
                </Button>
              }
            />
          </div>
        )}

        {result && (
          <div className="batch-result">
            <Divider />
            <Title level={4}>处理结果统计</Title>
            <div className="result-stats">
              <Paragraph>
                <strong>总计处理图片:</strong> {result.total} 张
              </Paragraph>
              <Paragraph>
                <strong>识别为猫的图片:</strong> {result.cat} 张
              </Paragraph>
              <Paragraph>
                <strong>识别为狗的图片:</strong> {result.dog} 张
              </Paragraph>
              <Paragraph>
                <strong>处理失败的图片:</strong> {result.errors} 张
              </Paragraph>
            </div>

            {result.details && result.details.length > 0 && (
              <div className="result-details">
                <Divider />
                <Title level={4}>图片分类详情</Title>
                <Table
                  dataSource={result.details}
                  columns={columns}
                  rowKey="filename"
                  pagination={{ pageSize: 10 }}
                />
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};

export default BatchProcessor;
