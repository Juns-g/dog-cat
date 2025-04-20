import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { corsMiddleware } from "./middleware/cors";
import { classifyRoute } from "./routes/classify";
import { batchClassifyRoute } from "./routes/batchClassify";

const app = new Hono();

app.use("/*", corsMiddleware);

classifyRoute(app);
batchClassifyRoute(app);

serve({
  fetch: app.fetch,
  port: 8787,
});

console.log("服务器启动成功");
