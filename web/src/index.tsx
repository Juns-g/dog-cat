/**
 * 猫狗图像分类系统的主组件
 * 提供图像上传、分类和批量处理功能
 */
import ImageClassifier from "./components/ImageClassifier";
import BatchProcessor from "./components/BatchProcessor";
import HistoryViewer from "./components/HistoryViewer";
import { Tabs } from "antd";
import Test from "./components/Test";

// 定义Tab项
const tabItems = [
  {
    key: "1",
    label: "单张图片分类",
    children: <ImageClassifier />,
  },
  {
    key: "2",
    label: "批量图片处理",
    children: <BatchProcessor />,
  },
  {
    key: "3",
    label: "分类历史记录",
    children: <HistoryViewer />,
  },
  {
    key: "4",
    label: "测试",
    children: <Test />,
  },
];

// 主组件
const CatDogClassifier: React.FC = () => {
  return (
    <div className="cat-dog-classifier">
      <Tabs defaultActiveKey="1" items={tabItems} />
    </div>
  );
};

export default CatDogClassifier;
