import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import { errorResponse } from "../utils/errorResponse";
import { Hono } from "hono";
import {
  BatchClassificationParams,
  BatchClassificationResult,
} from "../../shared/types";

const validateSchema = z.object({
  input_dir: z.string(),
  output_cat_dir: z.string().optional(),
  output_dog_dir: z.string().optional(),
});

export const batchClassifyRoute = (app: Hono) => {
  app.post(
    "/api/batch-classify",
    zValidator("json", validateSchema),
    async (c) => {
      try {
        const data: BatchClassificationParams = await c.req.json();
        const response = await fetch(
          "http://localhost:5001/api/batch-classify",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
          }
        );

        const result: BatchClassificationResult = await response.json();

        console.log("🚀 ~ py batch classify response:", result);

        return c.json(result);
      } catch (error: any) {
        console.error("批量分类过程中发生错误:", error);
        return errorResponse(c, "批量处理图片时出现内部错误", 500, {
          errorType: error.name,
          errorStack:
            process.env.NODE_ENV === "development" ? error.stack : undefined,
        });
      }
    }
  );
};
