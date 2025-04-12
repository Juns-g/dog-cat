import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { corsMiddleware } from "./middleware/cors";
import { classifyRoute } from "./routes/classify";
import { batchClassifyRoute } from "./routes/batchClassify";
import { historyRoute } from "./routes/history";

// 创建应用实例
const app = new Hono();

// 使用 CORS 中间件
app.use("/*", corsMiddleware);

// 注册路由
classifyRoute(app);
batchClassifyRoute(app);
historyRoute(app);
app.get("/", (c) => c.json({ message: "Hello World" }));

// 启动服务器
serve({
  fetch: app.fetch,
  port: 8787,
});

console.log("服务器启动成功");
