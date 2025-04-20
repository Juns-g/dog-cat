export type Class = "cat" | "dog";  

export interface ClassificationResult {
  success: boolean;
  class: Class;
  confidence: number;
}

