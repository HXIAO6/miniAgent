import os
import asyncio
import datetime
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, set_tracing_disabled

load_dotenv()
set_tracing_disabled(True)

base_url = os.getenv("OPENAI_BASE_URL")
if base_url:
    os.environ["OPENAI_API_BASE"] = base_url

@function_tool
def get_current_time() -> str:
    """获取当前的系统时间。当用户询问时间时，必须调用此工具。"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

my_agent = Agent(
    name="智能助手",
    model="deepseek-flash",
    instructions="你是一个严谨的助手。如果用户问时间，你必须调用工具查询。",
    tools=[get_current_time],
)

async def main():
    print("正在呼叫 DeepSeek (高级流式输出)...\n")
    
    # 问题测试
    result = Runner.run_streamed(my_agent, "现在几点了？请告诉我。顺便告诉我目前AI发展趋势")
    
    # 终端 ANSI 颜色代码
    GREY = "\033[90m"    # 灰色（用于思考）
    ITALIC = "\033[3m"   # 斜体（用于思考）
    CYAN = "\033[96m"    # 青色（用于最终回答）
    RESET = "\033[0m"    # 重置颜色
    
    is_thinking = False
    is_answering = False
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and hasattr(event.data, "delta"):
            e_type = getattr(event.data, "type", "")
            
            # 1. 处理思考过程（暗灰 + 斜体）
            if e_type in ["response.reasoning_summary_text.delta", "response.reasoning_text.delta"]:
                if not is_thinking:
                    # 如果之前已经开始回答了，突然又开始思考（罕见情况），先换行
                    if is_answering:
                        print("\n")
                    print(f"{GREY}{ITALIC}[🧠 思考中] ", end="")
                    is_thinking = True
                    is_answering = False
                # 输出灰色的文字
                print(f"{event.data.delta}", end="", flush=True)
                
            # 2. 处理最终回答（青色高亮）
            elif e_type == "response.output_text.delta":
                if not is_answering:
                    # 如果刚才在思考，结束思考的灰色样式
                    if is_thinking:
                        print(f"{RESET}\n\n") # 换个行，并重置颜色
                    print(f"[🤖 助手] {CYAN}", end="")
                    is_answering = True
                    is_thinking = False
                # 输出正常颜色的文字
                print(f"{event.data.delta}", end="", flush=True)

    print(f"{RESET}\n") # 最后重置一下颜色，防止影响终端

if __name__ == "__main__":
    asyncio.run(main())