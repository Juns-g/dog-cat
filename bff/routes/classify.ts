import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import { errorResponse } from "../utils/errorResponse";
import { Hono } from "hono";
import { isDev } from "../utils/env";
import { ClassificationResult } from "../../shared/types";

const validateSchema = z.object({
  image: z.string().optional(),
});

export const classifyRoute = (app: Hono) => {
  app.post("/api/classify", zValidator("json", validateSchema), async (c) => {
    try {
      const { image } = await c.req.json();

      if (!image) {
        return errorResponse(c, "没有提供图片数据", 400);
      }

      const response = await fetch("http://localhost:5001/api/classify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image }),
      });

      const data = await response.json();
      console.log("py classify", data);

      const result: ClassificationResult = {
        success: true,
        class: data.class,
        confidence: data.confidence,
      };

      return c.json(result);
    } catch (error: any) {
      console.error("/api/classify, 图片分类过程中发生错误:", error);
      return errorResponse(c, "处理图片时出现内部错误", 500, {
        errorType: error.name,
        errorStack: isDev ? error.stack : undefined,
      });
    }
  });
};
