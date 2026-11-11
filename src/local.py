import os
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv()

# ⚠️ 强制把本地地址注入环境变量，防止框架去连官方服务器
base_url = os.getenv("OPENAI_BASE_URL")
if base_url:
    os.environ["OPENAI_API_BASE"] = base_url

# 定义 Agent，注意这里 model 要写你本地有的模型
my_agent = Agent(
    name="本地小助手",
    model="qwen3.5:4b", # 注意这里要和 ollama 里的名字完全一致
    instructions="你是一个简洁的 Python 助手。",
)

print("正在呼叫本地大模型，首次响应可能有点慢，请稍候...\n")

# 运行
result = Runner.run_sync(my_agent, "告诉我你是谁")

print("\n--- 结果 ---")
print(result.final_output)