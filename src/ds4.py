import os
import asyncio
import datetime
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, set_tracing_disabled, ModelSettings
from tavily import TavilyClient

# 做出严格限制，严格调用工具，严格限制写入写出文件，工具调用可追踪
# ================= 1. 基础配置 =================
load_dotenv()
set_tracing_disabled(True)

base_url = os.getenv("OPENAI_BASE_URL")
if base_url:
    os.environ["OPENAI_API_BASE"] = base_url

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ================= 2. 定义工具 =================
@function_tool
def get_current_time() -> str:
    """获取当前的系统时间。当用户询问时间、日期时，必须调用此工具。"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

@function_tool
def search_web(query: str) -> str:
    """使用 Tavily 搜索引擎联网搜索最新信息。当需要查询实时新闻、未知知识时使用。"""
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        results = response.get("results", [])
        if not results:
            return "没有搜到相关结果。"
            
        formatted_results = ""
        for idx, res in enumerate(results):
            formatted_results += f"[{idx+1}] 标题: {res.get('title')}\n内容摘要: {res.get('content')}\n\n"
        return formatted_results
    except Exception as e:
        return f"搜索失败，错误信息: {e}"

ALLOWED_DIR = os.path.abspath("D:\\miniAgent")

def is_safe_path(file_path: str) -> bool:
    try:
        target_path = os.path.abspath(file_path)
        return target_path.startswith(ALLOWED_DIR)
    except Exception:
        return False

@function_tool
def read_local_file(file_path: str) -> str:
    """读取本地文件内容。"""
    if not is_safe_path(file_path):
        return "权限拒绝：只能访问 D:\\miniAgent 目录下的文件！"
    try:
        if os.path.getsize(file_path) > 500 * 1024: 
            return "文件太大，拒绝读取（超过 500KB）。"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"错误：找不到文件 '{file_path}'。"
    except Exception as e:
        return f"读取失败，错误信息: {e}"

@function_tool
def write_local_file(file_path: str, content: str) -> str:
    """将内容写入本地文件。"""
    if not is_safe_path(file_path):
        return "权限拒绝：只能在 D:\\miniAgent 目录下操作！"
    try:
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
    instructions="""你是一个严格执行指令的助手。规则如下：
    1. 绝对禁止凭空捏造事实。遇到最新信息，必须调用 search_web。
    2. 任何关于“现在时间”、“今天日期”的问题，必须调用 get_current_time，禁止自己编造时间。
    3. 遇到查看文件要求，调用 read_local_file。
    4. 遇到保存文件要求，调用 write_local_file。""",
    tools=[get_current_time, search_web, read_local_file, write_local_file],
    
    # 【新增 1】模型设置：强制要求调用工具
    # 默认是 "auto"（模型自己决定），改成 "required" 能极大减少它偷懒的情况
    model_settings=ModelSettings(
        temperature=0.1,      # 降低随机性，让它更严谨
        tool_choice="required" # 强制它必须调用工具（注意：这可能会导致它为了用工具而用工具，所以配合清晰的 instructions 最好）
    )
)

# ================= 4. 异步单次对话 =================
async def chat_once(conversation_history):
    GREY = "\033[90m"
    ITALIC = "\033[3m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m" # 新增：黄色，用来高亮显示工具调用
    MAGENTA = "\033[95m" # 新增：紫色，用来显示工具结果
    RESET = "\033[0m"
    
    result = Runner.run_streamed(my_agent, conversation_history)
    
    is_thinking = False
    is_answering = False
    assistant_reply = ""
    
    try:
        # 【新增 2】监听框架的高阶事件，专门捕获工具调用
        async for event in result.stream_events():
            # 处理文本生成
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                e_type = getattr(event.data, "type", "")
                
                if e_type in ["response.reasoning_summary_text.delta", "response.reasoning_text.delta"]:
                    if not is_thinking:
                        if is_answering:
                            print("\n")
                        print(f"{GREY}{ITALIC}[🧠 思考中] ", end="")
                        is_thinking = True
                        is_answering = False
                    print(f"{event.data.delta}", end="", flush=True)
                    
                elif e_type == "response.output_text.delta":
                    if not is_answering:
                        if is_thinking:
                            print(f"{RESET}\n\n", end="") 
                        print(f"[🤖 助手] {CYAN}", end="")
                        is_answering = True
                        is_thinking = False
                    
                    print(f"{event.data.delta}", end="", flush=True)
                    assistant_reply += event.data.delta

            # 【新增 3】监听工具调用的事件
            elif event.type == "run_item_stream_event":
                if event.name == "tool_called":
                    # 修复：通过 raw_item 访问工具名和参数
                    tool_name = getattr(event.item.raw_item, "name", "未知工具")
                    args = getattr(event.item.raw_item, "arguments", "无参数")
                    print(f"\n{YELLOW}  🔧 [正在调用工具]: {tool_name} 参数: {args}{RESET}")
                    is_answering = False # 如果之前有文本，强行打断换行
                    
                elif event.name == "tool_output":
                    # 当工具执行完毕，返回结果给模型时触发
                    tool_output = event.item.output
                    # 只截取前 100 个字符展示，防止终端刷屏
                    preview = str(tool_output)[:100].replace('\n', ' ') + "..."
                    print(f"{MAGENTA}  📥 [工具执行完毕]: {preview}{RESET}")
                    
        if assistant_reply:
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            
        print(f"{RESET}")
        
    except Exception as e:
        print(f"\n{RESET}❌ 发生错误: {e}")
        conversation_history.pop()

# ================= 5. 主循环 =================
def main():
    print("="*60)
    print("🤖 全能助手已启动！(输入 'quit' 退出)")
    print("="*60)
    
    conversation_history = []
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
        loop.run_until_complete(chat_once(conversation_history))

    loop.close()

if __name__ == "__main__":
    main()