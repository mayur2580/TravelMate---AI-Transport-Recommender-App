# 🚀 TravelMate - AI Transport Recommender App

TravelMate is an AI-powered assistant built with **LangGraph**, **Streamlit**, and **Groq LLaMA3** to help users find the best travel route (bus, train, or flight) between cities based on **cost**, **duration**, or **preference**. 🌍

### 🔧 Features
- Natural language input like “I want to go from Mumbai to Pune”
- Smart recommendation engine using LLM (Groq + LLaMA3)
- State memory using LangGraph 🧠
- Real-time route analytics 📊
- Route map visualization with Folium 🗺️
- Casual conversation support (e.g., travel talk or tourism)


### 🧠 Tech Stack
- LangGraph + LangChain
- Groq LLaMA3 70B
- Streamlit
- Pandas, Folium, dotenv

### 🛣️ Example Queries
- "I want to go from Chennai to Pune by train"
- "Which option is fastest from Delhi to Mumbai?"
- "Tell me something about traveling in India"

### 🚀 Run Locally
```bash
git clone https://github.com/yourusername/travelmate-ai.git
cd travelmate-ai
pip install -r requirements.txt
streamlit run app_copy.py

