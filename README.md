# Langchain Multi Agents

## Setup
Clone this repository
```

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




