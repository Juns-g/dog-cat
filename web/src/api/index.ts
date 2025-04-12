import axios from "axios";
import { ClassificationResult } from "../../../shared/types";

// 设置 API 域名
const API_DOMAIN = "http://localhost:8787"; // 更新为您的服务地址

// 创建一个axios实例
const axiosInstance = axios.create({
  baseURL: API_DOMAIN,
  withCredentials: true,
});

// 相应拦截器
axiosInstance.interceptors.response.use(
  (response) => {
    console.log("🚀 ~ response:", response);
    return response.data;
  },
  (error) => {
    console.log("🚀 ~ error:", error);
    return Promise.reject(error);
  }
);

// 发送图像分类请求
export const classifyImage = async (
  base64Image: string
): Promise<ClassificationResult> => {
  const response = await axiosInstance.post(`/api/classify`, {
    image: base64Image,
  });
  return response as any;
};

export const getTest = async () => {
  const response = await axiosInstance.get(`/`);
  return response;
};
