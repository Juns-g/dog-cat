import { ClassificationResult } from "../../shared/types";

export interface ClassifyResponse {
  success: boolean;
  result: ClassificationResult;
}