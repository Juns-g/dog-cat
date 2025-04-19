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
    console.log("🚀 ~ axios response:", response);
    return response.data;
  },
  (error) => {
    console.log("🚀 ~ axios error:", error);
    return Promise.reject(error);
  }
);

// 发送图像分类请求
export const classifyImage = async (
  base64Image: string
): Promise<ClassificationResult> => {
  // 检查图片大小
  const sizeInBytes = atob(base64Image.split(',')[1]).length;
  const maxSizeInMB = 5;
  
  if (sizeInBytes > maxSizeInMB * 1024 * 1024) {
    throw new Error(`图片大小不能超过 ${maxSizeInMB}MB`);
  }

  const response = await axiosInstance.post<any, ClassificationResult>(`/api/classify`, {
    image: base64Image,
  });
  console.log("🚀 ~ classifyImage response:", response)
  
  return response
};

export const getTest = async () => {
  const response = await axiosInstance.get(`/`);
  return response;
};
