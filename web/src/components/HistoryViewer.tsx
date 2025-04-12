/**
 * 分类历史记录查看组件
 * 显示之前所有的分类结果
 * 增强了错误处理和重试功能
 */
import React from "react";
import { Card, Table, Tag, Empty, Spin, Typography, Alert, Button } from "antd";
import { HistoryOutlined, ReloadOutlined } from "@ant-design/icons";
import useSWR from "swr";
import axios from "axios";
import { ClassificationHistory } from "../../../shared/types";

const { Title, Text } = Typography;

const fetcher = async (url: string) => {
  try {
    const response = await axios.get(url);
    return response.data;
  } catch (error) {
    // 提取错误详情
    let errorMessage = "加载历史记录失败";

    if (error.response) {
      const { data, status } = error.response;

      if (data && data.message) {
        errorMessage = data.message;
      }

      if (status >= 500) {
        // 服务器端错误，向开发运行时报告
        if (
          process.env.NODE_ENV === "development" &&
          typeof aipaDevRuntime !== "undefined"
        ) {
          aipaDevRuntime.reportApiError(
            {
              url: url,
              method: "GET",
              body: null,
            },
            errorMessage
          );
        }
      }
    } else if (error.request) {
      errorMessage = "无法连接到服务器，请检查网络连接或服务器状态。";
    } else {
      errorMessage = `请求错误: ${error.message}`;
    }

    throw new Error(errorMessage);
  }
};

const HistoryViewer: React.FC = () => {
  // 使用SWR获取历史记录
  const { data, error, isLoading, mutate } = useSWR(
    `${process.env.AIPA_API_DOMAIN}/api/history`,
    fetcher,
    {
      refreshInterval: 30000, // 每30秒刷新一次
      revalidateOnFocus: true, // 当页面获取焦点时重新验证
      shouldRetryOnError: true, // 出错时自动重试
      errorRetryCount: 3, // 最多重试3次
    }
  );

  // 手动刷新数据
  const handleRefresh = () => {
    mutate();
  };

  // 表格列定义
  const columns = [
    {
      title: "图片名称/路径",
      dataIndex: "imageUrl",
      key: "imageUrl",
      ellipsis: true,
    },
    {
      title: "分类结果",
      dataIndex: "classification",
      key: "classification",
      render: (text: string) => (
        <Tag color={text === "cat" ? "orange" : "green"}>
          {text === "cat" ? "🐱 猫" : "🐶 狗"}
        </Tag>
      ),
      filters: [
        { text: "猫", value: "cat" },
        { text: "狗", value: "dog" },
      ],
      onFilter: (value: string, record: any) => record.classification === value,
    },
    {
      title: "置信度",
      dataIndex: "confidence",
      key: "confidence",
      render: (confidence: number) => `${confidence}%`,
      sorter: (a: any, b: any) => a.confidence - b.confidence,
    },
    {
      title: "分类方法",
      dataIndex: "method",
      key: "method",
      render: (method: string) => (
        <Tag color={method === "api" ? "blue" : "purple"}>
          {method === "api" ? "单张分类" : "批量处理"}
        </Tag>
      ),
      filters: [
        { text: "单张分类", value: "api" },
        { text: "批量处理", value: "batch" },
      ],
      onFilter: (value: string, record: any) => record.method === value,
    },
    {
      title: "分类时间",
      dataIndex: "createdAt",
      key: "createdAt",
      render: (date: string) => new Date(date).toLocaleString(),
      sorter: (a: any, b: any) =>
        new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
      defaultSortOrder: "descend",
    },
  ];

  return (
    <div className="history-viewer">
      <Card
        title={
          <div className="history-title">
            <HistoryOutlined />
            <span style={{ marginLeft: 8 }}>分类历史记录</span>
          </div>
        }
        bordered={false}
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={handleRefresh}
            disabled={isLoading}
          >
            刷新
          </Button>
        }
      >
        {isLoading ? (
          <div className="loading-container">
            <Spin size="large" />
            <Text className="loading-text">正在加载历史记录...</Text>
          </div>
        ) : error ? (
          <div className="error-container">
            <Alert
              message="加载错误"
              description={error.message}
              type="error"
              showIcon
              action={
                <Button
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={handleRefresh}
                >
                  重试
                </Button>
              }
            />
          </div>
        ) : data && data.data && data.data.length > 0 ? (
          <Table
            dataSource={data.data}
            columns={columns}
            rowKey="_id"
            pagination={{
              pageSize: 10,
              showSizeChanger: true,
              pageSizeOptions: ["10", "20", "50"],
            }}
          />
        ) : (
          <Empty
            description="暂无分类历史记录"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        )}
      </Card>
    </div>
  );
};

export default HistoryViewer;
