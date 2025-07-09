import streamlit as st
import os
import json
import pandas as pd
import re
from dotenv import load_dotenv
import pydeck as pdk
from typing import TypedDict, Optional
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ========== Load Env & Data ==========
load_dotenv()
train_df = pd.read_csv("data/Ttrain_data.csv")
bus_df = pd.read_csv("data/Bus_data.csv")
plane_df = pd.read_csv("data/plane_data.csv")

for df in [train_df, bus_df, plane_df]:
    df["Source"] = df["Source"].astype(str).str.lower().str.strip()
    df["Destination"] = df["Destination"].astype(str).str.lower().str.strip()


llm = ChatGroq(
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# ========== Persistent Memory ==========
SESSION_FILE = "history/session_1.json"

def load_history():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return {"chat_history": [], "state_history": []}

def save_history(chat_history, state_history):
    os.makedirs("history", exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump({"chat_history": chat_history, "state_history": state_history}, f)

# ========== LangGraph State ==========
class AgentState(TypedDict):
    user_input: str
    source: Optional[str]
    destination: Optional[str]
    preference: Optional[str]
    result: Optional[str]
    chat_history: list

# ========== Nodes ==========
def input_node(state: AgentState) -> AgentState:
    query = state["user_input"]
    history = state.get("chat_history", [])

    full_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    prompt = f"""You are a helpful travel assistant. Based on this conversation:

{full_context}

User said: \"{query}\"

Classify intent first: 
- If it's a casual travel/tourism conversation, just respond naturally.
- If it contains a clear route query, extract:
  - Source city
  - Destination city
  - Preference (cost, time, best, or mode like 'train')


If it's casual talk, reply in this format:
Intent: casual
Response: <friendly reply>

If it's a travel query, reply in this format:
Intent: route
Source: <city>
Destination: <city>
Preference: <cost/time/best/train/bus/plane>
"""

    response = llm.invoke(prompt).content.lower()

    if "intent: casual" in response:
        match = re.search("response:\s*(.*)", response)
        return {**state, "result": match.group(1).strip() if match else "Let's continue our chat!"}

    if "missing:" in response:
        missing = response.split("missing:")[1].strip()
        return {**state, "result": f"🤖 Can you please tell me your {missing}?"}

    def extract_value(label):
        match = re.search(f"{label.lower()}\\s*:\\s*([\\w]+)", response)
        return match.group(1).strip() if match else None

    return {
        **state,
        "source": extract_value("source"),
        "destination": extract_value("destination"),
        "preference": extract_value("preference"),
    }

def get_best_transport(state: AgentState) -> AgentState:
    source = state["source"]
    destination = state["destination"]
    preference = (state["preference"] or "best").lower()
    valid_preferences = {"cost", "time", "best", "train", "bus", "plane"}

    if not source or not destination:
        return {**state, "result": "Please specify both source and destination cities."}

    if preference not in valid_preferences:
        preference = "best"

    results = []

    def safe_get(row, col):
        return row[col] if col in row and pd.notna(row[col]) else float("inf")

    def match(df, mode, src_col, dst_col, cost_col, time_col):
        matches = df[
            (df[src_col].str.lower().str.strip() == source.lower().strip()) &
            (df[dst_col].str.lower().str.strip() == destination.lower().strip())
        ]
        for _, row in matches.iterrows():
            cost = safe_get(row, cost_col)
            time = safe_get(row, time_col)
            if cost != float("inf") and time != float("inf"):
                results.append({
                    "mode": mode,
                    "cost": float(cost),
                    "time": float(time)
                })

    # Mode-specific preference
    preferred_mode = None
    if preference in {"train", "bus", "plane"}:
        preferred_mode = preference.capitalize()
        preference = "best"  # Revert to sorting logic

    # Match based on mode filter
    if preferred_mode in [None, "Bus"]:
        match(bus_df, "Bus", "Source", "Destination", "Cost", "Duration")
    if preferred_mode in [None, "Train"]:
        match(train_df, "Train", "Source", "Destination", "Cost", "Duration")
    if preferred_mode in [None, "Plane"]:
        match(plane_df, "Plane", "Source", "Destination", "Cost", "Duration")

    if not results:
        return {**state, "result": "❌ No transport found for this route."}

    # Sort result based on cost/time/best
    if preference == "cost":
        best = min(results, key=lambda x: x["cost"])
    elif preference == "time":
        best = min(results, key=lambda x: x["time"])
    else:
        best = min(results, key=lambda x: x["cost"] + x["time"])

    result = (
        f"✅ Best mode: {best['mode']}\n"
        f"💸 Cost: ₹{best['cost']:.2f}\n"
        f"🕒 Time: {best['time']:.2f} hours"
    )
    return {**state, "result": result}
    source = state["source"]
    destination = state["destination"]
    preference = (state["preference"] or "best").lower()
    valid_preferences = {"cost", "time", "best"}
    if preference not in valid_preferences:
        preference = "best"

    preferred_mode = None
    if preference in {"train", "bus", "plane"}:
        preferred_mode = preference.capitalize()
        preference = "best"  # fallback for sorting


    if not source or not destination:
        return {**state, "result": "Please specify both source and destination cities."}

    results = []

    def safe_get(row, col):
        return row[col] if col in row and pd.notna(row[col]) else float("inf")

    def match(df, mode, src_col, dst_col, cost_col, time_col):
        matches = df[
            (df[src_col].str.lower().str.strip() == source.lower().strip()) &
            (df[dst_col].str.lower().str.strip() == destination.lower().strip())
        ]
        for _, row in matches.iterrows():
            cost = safe_get(row, cost_col)
            time = safe_get(row, time_col)
            if cost != float("inf") and time != float("inf"):
                results.append({
                    "mode": mode,
                    "cost": float(cost),
                    "time": float(time)
                })

    preferred_mode = None
    if preference in {"train", "bus", "plane"}:
        preferred_mode = preference.capitalize()
        preference = "best"  # fallback for sorting purposes

# Now match only the requested mode (or all if none specified)
    if preferred_mode in [None, "Bus"]:
        match(bus_df, "Bus", "Source", "Destination", "Cost", "Duration")
    if preferred_mode in [None, "Train"]:
        match(train_df, "Train", "Source", "Destination", "Cost", "Duration")
    if preferred_mode in [None, "Plane"]:
        match(plane_df, "Plane", "Source", "Destination", "Cost", "Duration")

    if not results:
        return {**state, "result": "❌ No transport found for this route."}

    if preference == "cost":
        best = min(results, key=lambda x: x["cost"])
    elif preference == "time":
        best = min(results, key=lambda x: x["time"])
    else:
        best = min(results, key=lambda x: x["cost"] + x["time"])

    result = f"✅ Best mode: {best['mode']}\n💸 Cost: ₹{best['cost']:.2f}\n🕒 Time: {best['time']:.2f} hours"
    return {**state, "result": result}

def output_node(state: AgentState):
    return state

# ========== Graph ==========
graph = StateGraph(AgentState)
graph.add_node("input", input_node)
graph.add_node("recommend", get_best_transport)
graph.add_node("output", output_node)

graph.set_entry_point("input")
graph.add_conditional_edges("input", lambda state: END if "result" in state and ("🤖" in state["result"] or "✅" not in state["result"]) else "recommend")
graph.add_edge("recommend", "output")
graph.set_finish_point("output")

runnable = graph.compile()

# ========== Streamlit UI ==========
st.set_page_config(page_title="TravelMate Assistance ", layout="centered")
st.title("TravelMate Assistance")

if "chat_history" not in st.session_state:
    memory = load_history()
    st.session_state.chat_history = memory["chat_history"]
    st.session_state.state_history = memory["state_history"]

# Display chat
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# Analytics button
if st.sidebar.checkbox("📊 Show Analytics"):
    st.sidebar.markdown("### 📊 Trip Summary")
    df = pd.DataFrame(st.session_state.state_history)
    if not df.empty:
        trips = df.dropna(subset=["source", "destination"])
        st.sidebar.metric("Total Trips", len(trips))
        freq_routes = trips.groupby(["source", "destination"]).size().reset_index(name="count")
        st.sidebar.dataframe(freq_routes.sort_values("count", ascending=False).head(5))

# Map integration
import folium
from streamlit_folium import st_folium

# Map checkbox
if st.sidebar.checkbox("🗺️ Show Route Map"):
    coords_map = {
        "mumbai": [19.0760, 72.8777], "pune": [18.5204, 73.8567], "delhi": [28.6139, 77.2090],
        "bangalore": [12.9716, 77.5946], "hyderabad": [17.3850, 78.4867], "chennai": [13.0827, 80.2707]
    }

    latest = st.session_state.state_history[-1] if st.session_state.state_history else None
    if latest and latest.get("source") and latest.get("destination"):
        src_name = latest["source"].lower()
        dst_name = latest["destination"].lower()
        src = coords_map.get(src_name)
        dst = coords_map.get(dst_name)

        if src and dst:
            # Center the map between source and destination
            midpoint = [(src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2]
            route_map = folium.Map(location=midpoint, zoom_start=6)

            # Add markers
            folium.Marker(location=src, popup=src_name.capitalize(), icon=folium.Icon(color='green')).add_to(route_map)
            folium.Marker(location=dst, popup=dst_name.capitalize(), icon=folium.Icon(color='red')).add_to(route_map)

            # Add line between points
            folium.PolyLine(locations=[src, dst], color="blue", weight=4.5, opacity=0.8).add_to(route_map)

            # Display in Streamlit
            st.sidebar.subheader("🛣️ Route Preview")
            st_folium(route_map, width=400, height=300)
        else:
            st.sidebar.warning("⚠️ Route coordinates not found.")
    else:
        st.sidebar.info("ℹ️ No valid route available to show.")

# Rewind button
# if st.button("⏪ Rewind", use_container_width=True):
#     if len(st.session_state.state_history) >= 2:
#         st.session_state.state_history.pop()
#         last_state = st.session_state.state_history[-1]
#         st.session_state.chat_history = last_state.get("chat_history", [])
#         save_history(st.session_state.chat_history, st.session_state.state_history)
#         st.rerun()
#     else:
#         st.warning("⚠️ No previous state to rewind to.")

# Reset button
if st.button("🔄 Reset Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.state_history = []
    save_history([], [])
    st.rerun()

# Chat input
user_input = st.chat_input("Where do you want to go or what do you want to talk about?")
if user_input:
    with st.spinner("🤖 Thinking..."):
        history = st.session_state.chat_history
        state = {
            "user_input": user_input,
            "chat_history": history
        }
        final = runnable.invoke(state)

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": final["result"]})
        st.session_state.state_history.append(final)

        save_history(st.session_state.chat_history, st.session_state.state_history)
        st.chat_message("assistant").write(final["result"])
