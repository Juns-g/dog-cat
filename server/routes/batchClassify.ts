import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import { errorResponse } from "../utils/errorResponse";
import { Context } from "hono";

const batchClassifySchema = z.object({
  input_dir: z.string(),
  output_cat_dir: z.string().optional(),
  output_dog_dir: z.string().optional(),
});

export const batchClassifyRoute = (app: any) => {
  app.post(
    "/api/batch-classify",
    zValidator("json", batchClassifySchema),
    async (c: Context) => {
      try {
        const data = await c.req.json();
        // 处理批量分类逻辑...
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
