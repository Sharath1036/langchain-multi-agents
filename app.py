import asyncio
import streamlit as st
import tempfile
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.pdf_agent import PDFAgent
from agents.weather_agent import WeatherAgent

# Ensure an event loop exists for async libraries (fix for Google Generative AI Embeddings)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Langchain Agents Demo", layout="wide")
st.title("LangGraph Agents Demo")

tab1, tab2, tab3 = st.tabs(["PDF Agent", "Weather Agent", "Multi-Agent QA"])

with tab1:
    st.header("PDF Agent")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    question = st.text_input("Ask a question about the PDF:")
    if uploaded_pdf:
        if 'uploaded_pdf_data' not in st.session_state:
            st.session_state.uploaded_pdf_data = uploaded_pdf.read()
            st.session_state.uploaded_pdf_name = uploaded_pdf.name
            st.session_state.uploaded_pdf_size = uploaded_pdf.size


    if 'uploaded_pdf_name' in st.session_state and 'uploaded_pdf_size' in st.session_state:
        st.info(f"PDF uploaded: {st.session_state.uploaded_pdf_name}, size: {st.session_state.uploaded_pdf_size} bytes")

    if 'uploaded_pdf_data' in st.session_state and question:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(st.session_state.uploaded_pdf_data)
                tmp_path = tmp_file.name
            st.info(f"Saved PDF to temp file: {tmp_path}")
            pdf_agent = PDFAgent(pdf_path=tmp_path)
            with st.spinner("Processing..."):
                answer = pdf_agent.ask(question)
            st.success("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            import traceback
            st.text(traceback.format_exc())


with tab2:
    st.header("Weather Agent")
    location = st.text_input("Enter a location for weather info: e.g. Mumbai")
    if location:
        weather_agent = WeatherAgent()
        with st.spinner("Fetching weather..."):
            try:
                result = weather_agent.ask(location)
                st.success("Weather Info:")
                st.write(result)  # This might be None or a dict
                # Try to extract the answer if it's a dict or object
                # if isinstance(result, dict):
                #     # Try common keys
                #     if "output" in result:
                #         st.write(result["output"])
                #     elif "result" in result:
                #         st.write(result["result"])
                #     else:
                #         st.write(str(result))
                # elif hasattr(result, "content"):
                #     st.write(result.content)
                # elif result is not None:
                #     st.write(str(result))
            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.header("Multi-Agent QA (PDF + Weather)")
    user_input = st.text_area("Ask multiple questions (e.g. 'What organizations has Sharath worked for and tell me the weather in Mumbai'):")
    uploaded_pdf = st.file_uploader("Upload a PDF for PDF Agent (optional)", type=["pdf"], key="multi_pdf")
    if st.button("Ask Multi-Agent"):
        from nodes.node import split_questions, classify_question
        from langchain_core.messages import HumanMessage
        import tempfile
        messages = []
        # If PDF uploaded, save and use it
        pdf_path = None
        if uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                pdf_path = tmp_file.name
        # Split and process each question
        questions = split_questions(user_input)
        for question in questions:
            agent_name = classify_question(question)
            if agent_name == "pdf_agent":
                if pdf_path:
                    pdf_agent = PDFAgent(pdf_path=pdf_path)
                else:
                    pdf_agent = PDFAgent(pdf_path="Sharath_OnePage.pdf")
                result = pdf_agent.agent.invoke({"input": question})
                if isinstance(result, dict):
                    text_result = result.get("output") or result.get("text") or str(result)
                else:
                    text_result = str(result)
                messages.append(("PDF Agent", text_result))
            else:
                weather_agent = WeatherAgent()
                import re
                match = re.search(r"weather in ([\w\s,]+)", question, re.IGNORECASE)
                location = match.group(1).strip() if match else question
                result = weather_agent.ask(location)
                messages.append(("Weather Agent", str(result)))
        st.subheader("Results:")
        for agent, answer in messages:
            st.markdown(f"**{agent}:** {answer}")