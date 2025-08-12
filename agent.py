import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


if not os.getenv("LANGSMITH_API_KEY"):
    raise EnvironmentError("LANGSMITH_API_KEY not found in .env")
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")
if not os.getenv("TAVILY_API_KEY"):
    raise EnvironmentError("TAVILY_API_KEY not found in .env")

search = TavilySearch(max_results=3)
search_results = search.invoke("오늘 서울 날씨 알려줘")
tools = [search]

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
model_with_tools = model.bind_tools(tools)

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=memory,
)
config = {"configurable": {"thread_id": "abc123"}}


system = SystemMessage(
    content=(
        "너는 한국어로 간결하게 답한다. 웹 검색 도구를 사용할 수 있다면 사용하고, "
        "사용했다면 핵심 근거 URL을 한 줄로 요약해서 함께 제시한다."
    )
)

events = agent.stream(
    {"messages": [system, HumanMessage("오늘 날짜 알려줘")]},
    config,
    stream_mode="values",
)
for step in events:
    step["messages"][-1].pretty_print()

events = agent.stream(
    {"messages": [HumanMessage("오늘 서울 날씨 알려줘")]}, config, stream_mode="values"
)
for step in events:
    step["messages"][-1].pretty_print()

events = agent.stream(
    {"messages": [HumanMessage("내일 비 와? 출처도 알려줘")]},
    config,
    stream_mode="values",
)
for step in events:
    step["messages"][-1].pretty_print()
