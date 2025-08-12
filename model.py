import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
if not key:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

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
    response = model.invoke(state["messages"])
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
