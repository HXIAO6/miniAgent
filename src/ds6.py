import os
import asyncio
import datetime
import json
import gradio as gr
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, set_tracing_disabled, ModelSettings
from tavily import TavilyClient


#构建成浏览器页面
# ================= 1. 基础配置 =================
load_dotenv()
set_tracing_disabled(True)
    
base_url = os.getenv("OPENAI_BASE_URL")
if base_url:
    os.environ["OPENAI_API_BASE"] = base_url

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 定义历史文件路径 (完全采用你的修改)
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
history_dir = os.path.join(project_dir, "mem")
os.makedirs(history_dir, exist_ok=True)
HISTORY_FILE = os.path.join(history_dir, "history.json")

# ================= 2. 定义工具 =================
@function_tool
def get_current_time() -> str:
    """获取当前的系统时间。当用户询问时间、日期时，必须调用此工具。"""
    return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

@function_tool
def search_web(query: str) -> str:
    """使用 Tavily 搜索引擎联网搜索最新信息。"""
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        results = response.get("results", [])
        if not results: return "没有搜到相关结果。"
        formatted_results = ""
        for idx, res in enumerate(results):
            formatted_results += f"[{idx+1}] 标题: {res.get('title')}\n内容摘要: {res.get('content')}\n\n"
        return formatted_results
    except Exception as e:
        return f"搜索失败: {e}"

ALLOWED_DIR = os.path.abspath("D:\\miniAgent")

def is_safe_path(file_path: str) -> bool:
    try:
        return os.path.abspath(file_path).startswith(ALLOWED_DIR)
    except Exception:
        return False

@function_tool
def read_local_file(file_path: str) -> str:
    """读取本地文件内容。"""
    if not is_safe_path(file_path): return "权限拒绝！"
    try:
        if os.path.getsize(file_path) > 500 * 1024: return "文件太大，拒绝读取。"
        with open(file_path, "r", encoding="utf-8") as f: return f.read()
    except Exception as e:
        return f"读取失败: {e}"

@function_tool
def write_local_file(file_path: str, content: str) -> str:
    """将内容写入本地文件。"""
    if not is_safe_path(file_path): return "权限拒绝！"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f: f.write(content)
        return f"成功：内容已成功写入文件 '{file_path}'。"
    except Exception as e:
        return f"写入失败: {e}"

# ================= 3. 定义 Agent =================
my_agent = Agent(
    name="全能助手",
    model="deepseek-flash",
    instructions="""你是一个严格执行指令的助手。规则如下：
    1. 绝对禁止凭空捏造事实。遇到最新信息，必须调用 search_web。
    2. 任何关于“现在时间”的问题，必须调用 get_current_time，禁止自己编造时间。
    3. 遇到查看文件要求，调用 read_local_file。
    4. 遇到保存文件要求，调用 write_local_file。
    5. 遇到普通闲聊或询问历史记录，直接回答即可，不要乱用工具。""",
    tools=[get_current_time, search_web, read_local_file, write_local_file],
    model_settings=ModelSettings(temperature=0.1)
)

# ================= 4. 记忆压缩大师 =================
summarizer_agent = Agent(
    name="记忆压缩大师",
    model="deepseek-flash",
    instructions="""你是一个擅长总结的助手。请把用户提供的历史对话记录，浓缩成一段不超过 200 字的摘要。
    必须提取以下核心信息：
    1. 用户的身份或喜好。
    2. 用户要求你记住的特定内容。
    3. 之前任务的关键结果。
    直接输出摘要内容，不要废话。"""
)

async def compress_history(conversation_history):
    """如果历史记录太长，就调用这个函数进行压缩"""
    if len(conversation_history) > 12:
        print("\n⚙️ [系统后台] 对话太长，正在启动记忆压缩...")
        
        valid_messages = [msg for msg in conversation_history if msg['role'] != 'system']
        old_messages = valid_messages[:-4]
        recent_messages = valid_messages[-4:]
        
        history_text = ""
        for msg in old_messages:
            history_text += f"{msg['role']}: {msg['content']}\n"
            
        result = await Runner.run(summarizer_agent, f"请压缩以下对话：\n{history_text}")
        summary = result.final_output
        
        print(f"⚙️ [系统后台] 压缩完成！摘要内容：{summary[:30]}...")
        
        new_history = [
            {"role": "system", "content": f"【前情提要】: {summary}"}
        ] + recent_messages
        
        return new_history
    return conversation_history

# ================= 5. 网页端核心交互逻辑 =================
async def chat_generator(user_input, history_list):
    # 将网页格式转为 Agent 需要的格式
    conversation_history = [{"role": m["role"], "content": m["content"]} for m in history_list]
    conversation_history.append({"role": "user", "content": user_input})
    
    # 1. 后台静默压缩记忆
    conversation_history = await compress_history(conversation_history)
    
    # 2. 发起流式对话
    result = Runner.run_streamed(my_agent, conversation_history)
    
    current_reply = ""
    is_thinking = False
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and hasattr(event.data, "delta"):
            e_type = getattr(event.data, "type", "")
            
            # 处理思考过程
            if e_type in ["response.reasoning_summary_text.delta", "response.reasoning_text.delta"]:
                if not is_thinking:
                    current_reply += "\n\n> 🧠 **思考中**: "
                    is_thinking = True
                current_reply += event.data.delta
                
            # 处理最终回答
            elif e_type == "response.output_text.delta":
                if is_thinking:
                    current_reply += "\n\n"
                    is_thinking = False
                current_reply += event.data.delta
                
            yield current_reply
            
        # 处理工具调用
        elif event.type == "run_item_stream_event":
            if event.name == "tool_called":
                tool_name = getattr(event.item.raw_item, "name", "未知工具")
                current_reply += f"\n\n> 🔧 **正在调用工具**: `{tool_name}`\n\n"
                is_thinking = False
                yield current_reply

    # 对话结束，保存历史记录
    full_history = [{"role": m["role"], "content": m["content"]} for m in history_list]
    full_history.append({"role": "user", "content": user_input})
    full_history.append({"role": "assistant", "content": current_reply})
    
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(full_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存失败: {e}")

def predict(message, history):
    """包装成 Gradio 可识别的生成器"""
    # history 是列表，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    # 先追加用户消息
    history.append({"role": "user", "content": message})
    yield "", history
    
    # 准备接收回复
    history.append({"role": "assistant", "content": ""})
    
    # 运行异步生成器
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async_gen = chat_generator(message, history[:-2]) # 传入刚才的完整历史
    
    try:
        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                history[-1]["content"] = chunk
                yield "", history
            except StopAsyncIteration:
                break
    finally:
        loop.close()

# ================= 6. 加载历史记录 =================
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

# ================= 7. 构建网页界面 =================
with gr.Blocks(title="全能 AI 助手") as demo:
    gr.Markdown("# 🤖 全能 AI 助手 (网页版)")
    gr.Markdown("支持自动记忆、记忆压缩、联网搜索、本地文件读写。")
    
    chatbot = gr.Chatbot(height=600, value=load_history)
    msg = gr.Textbox(placeholder="输入你的问题，按回车发送...", label="输入框", scale=4)
    clear = gr.Button("清空记忆", variant="primary", scale=1)

    # 交互逻辑
    msg.submit(predict, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], []), None, [chatbot, chatbot], queue=False).then(
        lambda: os.remove(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else None
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=gr.themes.Soft())