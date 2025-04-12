export const errorResponse = (
  c: any,
  message: string,
  status: number = 500,
  details: any = null
) => {
  console.error(`Error (${status}): ${message}`, details ? details : "");
  return c.json(
    {
      error: true,
      message: message,
      details: details,
      timestamp: new Date().toISOString(),
    },
    status
  );
};
