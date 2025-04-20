export type Class = "cat" | "dog";

export interface ClassResult {
  filename?: string;
  class: Class;
  confidence: number;
}

export interface ClassificationResult extends ClassResult {
  success: boolean;
}

export interface BatchClassificationParams {
  input_dir: string;
  output_cat_dir?: string;
  output_dog_dir?: string;
}

export interface BatchClassificationResult {
  success: boolean;
  cat_count: number;
  dog_count: number;
  errors?: number;
  details?: ClassResult[];
}
