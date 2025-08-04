import os
import sys
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader


class PDFAgent:
    def __init__(self, pdf_path: str, collection_name: str = "test"):
        self.pdf_path = pdf_path
        self.collection_name = collection_name
        self._load_environment()
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._initialize_vector_store()
        self.qa_chain = self._initialize_qa_chain()
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent()

    def _load_environment(self):
        load_dotenv(override=True)
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY")
        os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL")
        os.environ["LANGSMITH_TRACING"]= "true"
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

    def _initialize_llm(self):
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.0,
            )
        
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429:
                print("Google Gemini rate limited. Falling back to Groq...")
            elif "429" in str(e):
                print("Rate limit detected in error message. Falling back to Groq...")
            else:
                raise  # re-raise other exceptions

            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.0,
            )

    def _initialize_embeddings(self):
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    def _initialize_vector_store(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_texts = text_splitter.split_documents(documents)

        return QdrantVectorStore.from_documents(
            documents=split_texts,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            api_key=os.getenv("QDRANT_API_KEY"),
            url=os.getenv("QDRANT_URL"),
            force_recreate=True
        )

    def _initialize_qa_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def _initialize_tools(self):
        tools = load_tools([], llm=self.llm)
        tools.append(
            Tool(
                name="State of Union QA System",
                func=self.qa_chain.run,
                description=(
                    "Useful for answering questions from the uploaded PDF. "
                    "Input should be a fully formed question."
                ),
            )
        )
        return tools

    def _initialize_agent(self):
        return initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def ask(self, question: str):
        print("Asking:", question)
        result = self.agent.run(question)
        print("Result:", result)
        return result



if __name__ == "__main__":
    print("Starting PDF Agent...")
    pdf_agent = PDFAgent(pdf_path="Sharath_OnePage.pdf")
    print("Agent initialized.")
    response = pdf_agent.ask("What all organizations has Sharath worked with?")
    print("Response:", response)
