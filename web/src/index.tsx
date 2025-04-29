import ImageClassifier from "./components/ImageClassifier";
import { Tabs } from "antd";

const tabItems = [
  {
    key: "1",
    label: "单张图片分类",
    children: <ImageClassifier />,
  },
];

const CatDogClassifier = () => (
  <div className="cat-dog-classifier">
    <Tabs defaultActiveKey="1" items={tabItems} />
  </div>
);

export default CatDogClassifier;
