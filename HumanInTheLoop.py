from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command 
from dotenv import load_dotenv

load_dotenv()

memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

@tool
def get_stock_price(symbol: str) -> float:
    '''
    Return the current price of a stock given the stock symbol
    :param symbol: stock symbol
    :return: current price of the stock
    '''
    return {
        "MSFT": 200.34,
        "AAPL": 190.2,
        "AMZN": 89.9
    }.get(symbol, 0.0)

@tool
def buy_stocks(symbol:str, quantity:int, total_price:float) -> str:
    '''buy stocks given the stock symbol and quantity'''

    decision = interrupt(f"Approve buying {quantity} {symbol} stocks for ${total_price:.2f}?")

    if decision == "yes":  
        return f"you bought {quantity} shares of {symbol} for a total price of {total_price}"
    else:
        return "buying declined"

tools = [get_stock_price, buy_stocks]
model_with_tools = model.bind_tools(tools)

def chatbot(state: State) -> State:
    return {"messages": [model_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot_node")
builder.add_conditional_edges("chatbot_node", tools_condition)
builder.add_edge("tools", "chatbot_node")
# builder.add_edge("chatbot_node", END)

graph = builder.compile(checkpointer=memory)

config = {'configurable': {'thread_id': '2'}}

state = graph.invoke({"messages": [{"role": "user", "content": "what is the current price of 10 AAPL?"}]}, config=config)
print(state["messages"][-1].content)

state = graph.invoke({"messages": [{"role": "user", "content": "buy 20 AAPL stock at current price."}]}, config=config)
print(state.get("__interrupt__"))

decision = input("Approve (yes/no): ")
state = graph.invoke(Command(resume=decision), config=config)
print(state["messages"][-1].content)