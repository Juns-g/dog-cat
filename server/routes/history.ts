import { errorResponse } from "../utils/errorResponse";
import { Context } from "hono";

export const historyRoute = (app: any) => {
  app.get("/api/history", async (c: Context) => {
    try {
      // 获取历史记录逻辑...
    } catch (error: any) {
      console.error("获取历史记录时发生错误:", error);
      return errorResponse(c, "获取历史记录失败", 500, {
        errorType: error.name,
        collection: "6cdd8cfc_classification_history",
      });
    }
  });
};
