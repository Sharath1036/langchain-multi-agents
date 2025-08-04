from typing import List, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from weather_agent import WeatherAgent
from pdf_agent import PDFAgent
from langgraph.graph import StateGraph, START
from IPython.display import Image, display
import re

def split_questions(user_message: str) -> List[str]:
    # Naive split on ' and ', ' then ', case insensitive
    parts = re.split(r'\band then\b|\band\b|\bthen\b', user_message, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]

def classify_question(question: str) -> Literal["pdf_agent", "weather_agent"]:
    # Simple keyword-based classification
    if re.search(r'\bweather\b', question, re.IGNORECASE):
        return "weather_agent"
    else:
        return "pdf_agent"

def pdf_agent_node(state: MessagesState) -> Command[Literal["weather_agent", END]]:
    pdf_agent = PDFAgent(pdf_path="Sharath_OnePage.pdf")
    user_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            user_message = message.content
            break
    if user_message is None:
        raise ValueError("No user message found in state.")

    result = pdf_agent.agent.invoke({"input": user_message})
    # Extract string from result
    if isinstance(result, dict):
        # Try common keys
        text_result = result.get("output") or result.get("text") or str(result)
    else:
        text_result = str(result)

    final_msg = HumanMessage(content=text_result, name="pdf_agent")
    goto = get_next_node(final_msg, "weather_agent")
    return Command(
        update={"messages": state["messages"] + [final_msg]},
        goto=goto,
    )

def weather_agent_node(state: MessagesState) -> Command[Literal["pdf_agent", END]]:
    weather_agent = WeatherAgent()
    user_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            user_message = message.content
            break
    if user_message is None:
        raise ValueError("No user message found in state.")

    match = re.search(r"weather in ([\w\s,]+)", user_message, re.IGNORECASE)
    location = match.group(1).strip() if match else user_message
    result = weather_agent.ask(location)
    final_msg = HumanMessage(content=result, name="weather_agent")
    goto = get_next_node(final_msg, "pdf_agent")
    return Command(
        update={"messages": state["messages"] + [final_msg]},
        goto=goto,
    )

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto

def build_graph():
    workflow = StateGraph(MessagesState)
    workflow.add_node("pdf_agent", pdf_agent_node)
    workflow.add_node("weather_agent", weather_agent_node)

    workflow.add_edge(START, "pdf_agent")
    workflow.add_edge("pdf_agent", "weather_agent")
    workflow.add_edge("weather_agent", END)

    graph = workflow.compile()
    return graph

if __name__ == "__main__":
    graph = build_graph()
    display(Image(graph.get_graph().draw_mermaid_png()))

    # Full user input with multiple questions
    user_input = "What organizations has Sharath worked for and tell me the weather in Mumbai"

    # Split into sub-questions
    questions = split_questions(user_input)

    # Prepare empty message list to accumulate conversation
    messages = []

    # Process each question routed to the correct agent node
    for question in questions:
        agent_name = classify_question(question)
        # Run the corresponding node manually with current messages + new question
        state = {"messages": messages + [HumanMessage(content=question)]}
        if agent_name == "pdf_agent":
            cmd = pdf_agent_node(state)
        else:
            cmd = weather_agent_node(state)

        # Update messages with agent response
        messages = cmd.update["messages"]

    # Print all agent responses
    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        print(f"{msg.name or 'user'}: {msg.content}")
