import os
import asyncio
import datetime
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, set_tracing_disabled

#能对话、能记忆、能调工具、流式输出

# ================= 1. 基础配置 =================
load_dotenv()
set_tracing_disabled(True)

base_url = os.getenv("OPENAI_BASE_URL")
if base_url:
    os.environ["OPENAI_API_BASE"] = base_url

# ================= 2. 定义工具 =================
@function_tool
def get_current_time() -> str:
    """获取当前的系统时间。当用户询问时间时，必须调用此工具。"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

# ================= 3. 定义 Agent =================
my_agent = Agent(
    name="智能助手",
    model="deepseek-flash",
    instructions="你是一个严谨、聪明的助手。如果用户问时间，必须调用工具查询。",
    tools=[get_current_time],
)

# ================= 4. 交互与流式输出主逻辑 =================
async def main():
    print("="*60)
    print("🤖 智能助手已启动！(输入 'quit' 或 'exit' 退出)")
    print("="*60)
    
    # 【核心】这就是 Agent 的“记忆”。它是一个列表，保存所有历史对话。
    conversation_history = []
    
    # 终端颜色配置
    GREY = "\033[90m"
    ITALIC = "\033[3m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    
    while True:
        # 获取用户输入
        user_input = input("\n👤 你: ")
        
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("👋 再见！")
            break
            
        if not user_input.strip():
            continue
            
        # 1. 把用户的输入存入记忆
        conversation_history.append({"role": "user", "content": user_input})
        
        # 2. 开始流式请求（注意：这里传的是 conversation_history，而不是单纯的一句话）
        result = Runner.run_streamed(my_agent, conversation_history)
        
        is_thinking = False
        is_answering = False
        assistant_reply = "" # 用来收集这次回答的完整文本，以便存入记忆
        
        try:
            # 3. 监听流式事件
            async for event in result.stream_events():
                if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                    e_type = getattr(event.data, "type", "")
                    
                    # --- A. 处理思考过程 ---
                    if e_type in ["response.reasoning_summary_text.delta", "response.reasoning_text.delta"]:
                        if not is_thinking:
                            if is_answering:
                                print("\n")
                            print(f"{GREY}{ITALIC}[🧠 思考中] ", end="")
                            is_thinking = True
                            is_answering = False
                        print(f"{event.data.delta}", end="", flush=True)
                        
                    # --- B. 处理最终回答 ---
                    elif e_type == "response.output_text.delta":
                        if not is_answering:
                            if is_thinking:
                                print(f"{RESET}\n\n", end="") # 结束思考，换行并重置颜色
                            print(f"[🤖 助手] {CYAN}", end="")
                            is_answering = True
                            is_thinking = False
                        
                        print(f"{event.data.delta}", end="", flush=True)
                        assistant_reply += event.data.delta # 收集回答
                        
            # 4. 把模型这次的回答也存入记忆，供下一轮使用
            if assistant_reply:
                conversation_history.append({"role": "assistant", "content": assistant_reply})
                
        except Exception as e:
            print(f"\n{RESET}❌ 发生错误: {e}")
            # 如果出错，把刚才用户的话从记忆里删掉，防止记忆错乱
            conversation_history.pop()

    print(f"{RESET}") # 退出时重置终端颜色

if __name__ == "__main__":
    asyncio.run(main())