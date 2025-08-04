import os
import sys
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import AgentType, Tool, initialize_agent

class WeatherAgent:
    def __init__(self):
        self._load_environment()
        self.weather_tool = self._initialize_weather_tool()
        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent()

    def _load_environment(self):
        load_dotenv(override=True)
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")
        os.environ["LANGSMITH_TRACING"]= "true"
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

    def _initialize_weather_tool(self):
        return OpenWeatherMapAPIWrapper()

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
                raise 

            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.0,
            )


    def _initialize_tools(self):
        return [
            Tool(
                name="weather",
                func=self.weather_tool.run,
                description="Use this tool to get the current weather in a specified location."
            )
        ]
    
    def _initialize_agent(self):
        return initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def ask(self, location: str):
        prompt = f"What's the weather like in {location}?"
        print("Asking:", prompt)
        result = self.agent.run(prompt)
        print("Result:", result)
        return result

if __name__ == "__main__":
    print("Starting Weather Agent...")
    weather_agent = WeatherAgent()
    print("Agent initialized.")
    location = "Avignon"  # Example location
    response = weather_agent.ask(location)
    print("Response:", response)
