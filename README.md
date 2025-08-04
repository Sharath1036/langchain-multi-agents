# Langchain Multi Agents

## Softwares
- OS: `Windows 11`
- Python: `3.11.0`
- pip: `22.3`

## Setup
Clone this repository
```
git clone https://github.com/Sharath1036/langchain-multi-agents.git
```

Creating a virtual environment
```
python -m venv myenv
```
```
myenv\Scripts\activate
```

Adding environment variables. Create a `.env` file add add the below variables
```
GROQ_API_KEY
OPENWEATHERMAP_API_KEY
QDRANT_API_KEY
QDRANT_URL
LANGSMITH_API_KEY
GOOGLE_API_KEY
```

## Running the code 
### Running the PDF Agent
```
python agents/pdf_agent.py
```

### Running the Weather Agent
```
python agents/weather_agent.py
```

### Running the nodes
```
python nodes/node.py
```

### Running the streamlit application
```
streamlit run app.py
```




