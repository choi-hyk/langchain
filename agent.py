import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ----------------------------
# 0) ENV
# ----------------------------
load_dotenv()

if not os.getenv("LANGSMITH_API_KEY"):
    raise EnvironmentError("LANGSMITH_API_KEY not found in .env")
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

# ----------------------------
# 1) 연산 도구 정의 (AST 기반 안전 계산)
# ----------------------------
import ast
import operator as op
from langchain_core.tools import tool

_ALLOWED = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Num):  # Py3.8-
        return node.n
    if isinstance(node, ast.Constant):  # int/float만 허용
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only int/float constants allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Expr):
        return _safe_eval(node.value)
    raise ValueError(f"disallowed syntax: {type(node).__name__}")


@tool("calc", return_direct=False)
def calc(expression: str) -> str:
    """
    사칙연산, 거듭제곱, 나머지, 몫을 계산한다
    허용: + - * / // % ** 와 괄호, 실수/정수
    예: 2*(3+4)**2 // 3
    """
    try:
        # ^ 를 ** 로 자동 치환(LLM이 ^를 사용할 수 있어 보정)
        expr = expression.replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        value = _safe_eval(tree.body)
        return f"{value}"
    except Exception as e:
        return f"error={type(e).__name__}: {e}"


# ----------------------------
# 2) 모델 & 에이전트
# ----------------------------
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0,
    max_output_tokens=256,
)

tools = [calc]  # 날씨/검색 도구 제거 → 연산 도구만 사용

memory = MemorySaver()
agent = create_react_agent(model=model, tools=tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

system = SystemMessage(
    content=(
        "너는 한국어로 간결하게 답한다 "
        "수식이 포함된 질문에서만 calc 도구를 사용하고, 도구에는 '순수 수식'만 전달한다 "
        "불확실하거나 수식이 아니면 도구를 쓰지 말고 솔직히 모른다고 답한다"
    )
)


# ----------------------------
# 3) 히스토리 트리밍 & 라우팅
# ----------------------------
def trim(history):
    return trim_messages(
        history,
        max_tokens=1200,
        token_counter=model,
        include_system=True,
        start_on="human",
    )


def need_calc(q: str) -> bool:
    # 수학 수식/연산 키워드/연산자 존재 여부로 간단 라우팅
    symbols = ["+", "-", "*", "/", "//", "%", "**", "^", "(", ")"]
    keywords = [
        "계산",
        "값",
        "더해",
        "빼",
        "곱",
        "나눠",
        "제곱",
        "거듭제곱",
        "몫",
        "나머지",
    ]
    return any(s in q for s in symbols) or any(k in q for k in keywords)


history = [system]


def ask(q: str):
    msgs = trim(history + [HumanMessage(q)])
    if need_calc(q):
        last = None
        for step in agent.stream({"messages": msgs}, config, stream_mode="values"):
            last = step
        return last["messages"][-1].content
    else:
        return model.invoke(msgs).content


# ----------------------------
# 4) 테스트
# ----------------------------
print(ask("2*(3+4)**2 // 3 값을 계산해줘"))  # calc 사용
print(ask("3^2 + 4^2 의 값은?"))  # ^ 자동 보정 → calc 사용
print(ask("파이 값 알려줘"))  # 수식 아님 → 모델 직답
