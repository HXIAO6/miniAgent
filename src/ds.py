import os
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, set_tracing_disabled, set_default_openai_api
import datetime

load_dotenv()
set_tracing_disabled(True)  # 关掉追踪

# 中转站，加上这行强制用传统 Chat API
# set_default_openai_api("chat_completions") 

base_url = os.getenv("OPENAI_BASE_URL")
if base_url:
    os.environ["OPENAI_API_BASE"] = base_url

# 1. 定义一个工具（注意 @function_tool 装饰器）
@function_tool
def get_current_time() -> str:
    """获取当前的系统时间。当用户询问时间时，必须调用此工具。"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

# 2. 定义 Agent，并把工具交给它
my_agent = Agent(
    name="智能助手",
    model="deepseek-flash",  # 确保这里是你中转站能用的模型名
    instructions="你是一个严谨的助手。如果用户问时间，你必须调用工具查询，绝对不能自己瞎猜。",
    tools=[get_current_time], # 挂载工具
)

print("正在呼叫 DeepSeek 并测试工具调用...\n")

# 3. 提一个必须用工具才能回答的问题
result = Runner.run_sync(my_agent, "现在几点了？")

print("\n--- 最终回答 ---")
print(result.final_output)