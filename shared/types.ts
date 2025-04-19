/**
 * 共享类型定义，供前端和服务端使用
 */

// 分类结果类型
export interface ClassificationResult {
  success: boolean;
  class: string;
  confidence: number;
  imageUrl?: string;
  id?: string;
  createdAt?: string;
  updatedAt?: string;
}


// 批量分类结果类型
export interface BatchClassificationResult {
  total: number;
  cat: number;
  dog: number;
  errors: number;
  details: {
    filename: string;
    classification: string;
    confidence: number;
  }[];
}

// 分类历史记录类型
export interface ClassificationHistory {
  _id: string;
  imageUrl: string;
  classification: string;
  confidence: number;
  method: string;
  createdAt: string;
  updatedAt: string;
}
