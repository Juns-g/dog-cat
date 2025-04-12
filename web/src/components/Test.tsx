import { getTest } from "@/api";
import { useMount } from "ahooks";

const Test = () => {
  useMount(async () => {
    const res = await getTest();
    console.log("🚀 ~ useMount ~ res:", res);
  });
  return <div>Test</div>;
};

export default Test;
