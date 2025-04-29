/**
 * 猫狗图像分类系统的主组件
 * 提供图像上传、分类和批量处理功能
 */
import ImageClassifier from "./components/ImageClassifier";
import { Tabs } from "antd";

const tabItems = [
  {
    key: "1",
    label: "单张图片分类",
    children: <ImageClassifier />,
  },
];

const CatDogClassifier: React.FC = () => {
  return (
    <div className="cat-dog-classifier">
      <Tabs defaultActiveKey="1" items={tabItems} />
    </div>
  );
};

export default CatDogClassifier;
