import axios from "axios";
import { ClassificationResult } from "../../../shared/types";

const API_DOMAIN = "http://localhost:8787"; 

const axiosInstance = axios.create({
  baseURL: API_DOMAIN,
  withCredentials: true,
});

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

export const classifyImage = async (
  base64Image: string
): Promise<ClassificationResult> => {
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
