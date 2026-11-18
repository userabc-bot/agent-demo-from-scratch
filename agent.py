"""一个最小可运行的 Agent Demo：基于 LangGraph 的工具调用循环。

运行前请在 .env 中配置 OPENAI_API_KEY（支持 DeepSeek / 通义千问等 OpenAI 兼容接口）。
"""

import operator
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

load_dotenv()

# ---------- 1. 模型 ----------
# DeepSeek 默认地址；换成通义千问时把 MODEL_NAME/MODEL_BASE_URL 改成 qwen-plus
# 和 https://dashscope.aliyuncs.com/compatible-mode/v1 即可
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    base_url=os.getenv("MODEL_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)


# ---------- 2. 工具 ----------
@tool
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数之积。"""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """计算 a 除以 b，b 不能为 0。"""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


# ---------- 3. 状态 ----------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# ---------- 4. 节点：调用模型 ----------
SYSTEM_PROMPT = "你是一个算术助手。遇到算术问题时必须调用工具计算，不要心算。"


def llm_call(state: MessagesState):
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------- 5. 节点：执行工具 ----------
def tool_node(state: MessagesState):
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        fn = tools_by_name[tool_call["name"]]
        observation = fn.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": results}


# ---------- 6. 路由：继续调用工具还是结束 ----------
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last_message = state["messages"][-1]
    return "tool_node" if last_message.tool_calls else END


# ---------- 7. 组装图 ----------
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [HumanMessage(content="先算 12 乘以 8，再加上 4，最后除以 5。")]},
        config={"recursion_limit": 10},
    )
    for msg in result["messages"]:
        msg.pretty_print()
    print(f"\n共调用模型 {result['llm_calls']} 次")
