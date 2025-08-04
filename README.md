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

## Screenshots
### PDF Agent
<img width="1365" height="630" alt="image" src="https://github.com/user-attachments/assets/13e0ca46-cd8e-42a1-aa97-462591f34196" />

### Weather Agent
<img width="1365" height="627" alt="image" src="https://github.com/user-attachments/assets/48c87129-8997-470e-92ad-301427bece38" />

### PDF Node
<img width="1364" height="630" alt="image" src="https://github.com/user-attachments/assets/7ca27be9-10ee-49ef-b60a-8a79a8b44069" />

### Weather Node
<img width="1365" height="632" alt="image" src="https://github.com/user-attachments/assets/956c797a-1fce-4dce-b3a4-0e818347669c" />


### Qdrant Vector Embeddings
<img width="1365" height="630" alt="image" src="https://github.com/user-attachments/assets/6a6b6ca7-48aa-4da4-bcd0-44fa5535847e" />





