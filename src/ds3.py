import os
import asyncio
import datetime
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, set_tracing_disabled
from tavily import TavilyClient


# 联网搜索功能 ， 读写文件功能 ，但未加限制
# ================= 1. 基础配置 =================
load_dotenv()
set_tracing_disabled(True)

base_url = os.getenv("OPENAI_BASE_URL")
if base_url:
    os.environ["OPENAI_API_BASE"] = base_url

# 初始化搜索客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ================= 2. 定义工具 =================

@function_tool
def get_current_time() -> str:
    """获取当前的系统时间。当用户询问时间时，必须调用此工具。"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

@function_tool
def search_web(query: str) -> str:
    """使用 Tavily 搜索引擎联网搜索最新信息。当需要查询实时新闻、未知知识时使用。
    
    Args:
        query: 搜索关键词，最好使用精炼的短句。
    """
    try:
        # 使用 Tavily 搜索，最多返回 3 条结果以节省 Token
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        results = response.get("results", [])
        
        if not results:
            return "没有搜到相关结果。"
            
        # 把搜索结果拼接成文本发给大模型
        formatted_results = ""
        for idx, res in enumerate(results):
            formatted_results += f"[{idx+1}] 标题: {res.get('title')}\n内容摘要: {res.get('content')}\n\n"
        return formatted_results
    except Exception as e:
        return f"搜索失败，错误信息: {e}"

@function_tool
def read_local_file(file_path: str) -> str:
    """读取本地文件内容。当用户要求查看代码、读取文本时需要。

    
    Args:
        file_path: 文件的相对路径或绝对路径。
    """
    try:
        # 简单安全防护：防止读取过大文件
        if os.path.getsize(file_path) > 500 * 1024: # 大于 500KB
            return "文件太大，拒绝读取（超过 500KB）。"
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"错误：找不到文件 '{file_path}'，请检查路径是否正确。"
    except Exception as e:
        return f"读取失败，错误信息: {e}"

@function_tool
def write_local_file(file_path: str, content: str) -> str:
    """将内容写入本地文件。如果文件已存在则覆盖，不存在则创建。
    
    Args:
        file_path: 要写入的文件路径。
        content: 要写入的文本内容。
    """
    try:
        # 确保文件夹存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功：内容已成功写入文件 '{file_path}'。"
    except Exception as e:
        return f"写入失败，错误信息: {e}"

# ================= 3. 定义 Agent =================
my_agent = Agent(
    name="全能助手",
    model="deepseek-flash",
    instructions="""你是一个拥有强大工具的全能助手。
    1. 遇到不知道的最新信息，请使用 search_web 查询。
    2. 遇到用户要求查看文件，请使用 read_local_file。
    3. 遇到用户要求写代码并保存，请使用 write_local_file。
    4. 遇到问时间，使用 get_current_time。
    请自主规划步骤，不要瞎编乱造。""",
    tools=[get_current_time, search_web, read_local_file, write_local_file],
)

# ================= 4. 异步单次对话（修复了 input 阻塞的问题） =================
async def chat_once(conversation_history):
    GREY = "\033[90m"
    ITALIC = "\033[3m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    
    result = Runner.run_streamed(my_agent, conversation_history)
    
    is_thinking = False
    is_answering = False
    assistant_reply = ""
    
    try:
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
                            print(f"{RESET}\n\n", end="") 
                        print(f"[🤖 助手] {CYAN}", end="")
                        is_answering = True
                        is_thinking = False
                    
                    print(f"{event.data.delta}", end="", flush=True)
                    assistant_reply += event.data.delta
                    
        if assistant_reply:
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            
        print(f"{RESET}")
        
    except Exception as e:
        print(f"\n{RESET}❌ 发生错误: {e}")
        conversation_history.pop()

# ================= 5. 主循环（同步） =================
def main():
    print("="*60)
    print("🤖 全能助手已启动！(输入 'quit' 退出)")
    print("="*60)
    
    conversation_history = []
    
    # 【核心修复】创建一个持久的事件循环，不要用 asyncio.run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("👋 再见！")
            break
        if not user_input.strip():
            continue
            
        conversation_history.append({"role": "user", "content": user_input})
        
        # 用这个常驻的 loop 去跑单次对话，而不是 asyncio.run
        loop.run_until_complete(chat_once(conversation_history))

    # 退出时，再优雅地关闭循环
    loop.close()

if __name__ == "__main__":
    main()