import os
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if not key:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "세 명의 다른 전문가들이 내가 하는 질문에 답하고 있다고 상상해보도록 해."
            "모든 전문가들은 자신의 생각의 한 단계를 적어내고,그것을 그룹과 공유할거야."
            "그런 다음 모든 전문가들은 다음 단계로 넘어가. "
            "만약 어떤 전문가가 어떤 시점에서든 자신이 틀렸다는 것을 깨닫게 되면 그들은 떠나고 마지막에 남은 답변을 제공해.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

query = "골프의 목적 중 하나는 다른 사람보다 더 높은 점수를 얻기 위해 노력하는 것이다. 예, 아니오?"
input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages},
    config,
)
output["messages"][-1].pretty_print()
