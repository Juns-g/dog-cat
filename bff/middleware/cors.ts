import { cors } from "hono/cors";

export const corsMiddleware = cors({
  origin: "http://localhost:3000",
  allowHeaders: [
    "X-Custom-Header",
    "Upgrade-Insecure-Requests",
    "Content-Type",
  ],
  allowMethods: ["POST", "GET", "OPTIONS"],
  exposeHeaders: ["Content-Length", "X-Kuma-Revision"],
  maxAge: 600,
  credentials: true,
});
