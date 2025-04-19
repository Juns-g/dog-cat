import React from "react";
import { Layout, Typography } from "antd";
import CatDogClassifier from "./index";
import "./styles/app.css";

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

const App: React.FC = () => {
  return (
    <Layout className="app-layout">
      <Header className="app-header">
        <Title level={2} className="header-title">
          基于 MobileNetV2 的猫狗图像分类系统
        </Title>
      </Header>
      <Content className="app-content">
        <div className="content-container">
          <CatDogClassifier />
        </div>
      </Content>
      <Footer className="app-footer">
        猫狗图像分类系统 &copy; {new Date().getFullYear()}
      </Footer>
    </Layout>
  );
};

export default App;
