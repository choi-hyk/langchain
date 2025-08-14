import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

gemini_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
openai_model = init_chat_model("gpt-4o-mini", model_provider="openai")


# resp = model.invoke(
#     [
#         HumanMessage(content="Hello, my name is choihyeok"),
#         AIMessage(content="Hello choihyeok! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]
# )
# print(resp.content)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    response = openai_model.invoke(state["messages"])
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm HYK."
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

query = "What's my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc234"}}

query = "What's my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc123"}}

query = "What's my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
