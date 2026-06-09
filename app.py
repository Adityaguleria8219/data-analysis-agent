import streamlit as st
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import StringIO
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

st.set_page_config(page_title="Data Analysis Agent", page_icon="⚡", layout="wide")
st.title("⚡ Data Analysis Agent")
st.caption("Powered by Gemini + Python")

CSV_DATA = """date,product,category,region,units_sold,revenue,cost
2024-01-05,Widget A,Electronics,North,120,2400,1200
2024-01-08,Widget B,Electronics,South,85,1700,850
2024-01-12,Gadget X,Home,East,200,4000,1600
2024-01-15,Gadget Y,Home,West,150,3000,1200
2024-01-20,Widget A,Electronics,East,95,1900,950
2024-01-22,Gadget X,Home,North,175,3500,1400
2024-02-02,Widget B,Electronics,West,110,2200,1100
2024-02-05,Gadget Z,Outdoor,South,60,1800,900
2024-02-10,Widget A,Electronics,South,130,2600,1300
2024-02-14,Gadget Y,Home,East,180,3600,1440
2024-02-18,Gadget Z,Outdoor,North,45,1350,675
2024-02-22,Widget B,Electronics,East,95,1900,950
2024-03-01,Widget A,Electronics,West,140,2800,1400
2024-03-05,Gadget X,Home,South,220,4400,1760
2024-03-10,Gadget Z,Outdoor,East,80,2400,1200
2024-03-15,Widget B,Electronics,North,105,2100,1050
2024-03-20,Gadget Y,Home,West,160,3200,1280
2024-03-25,Widget A,Electronics,South,115,2300,1150
2024-04-01,Gadget X,Home,North,195,3900,1560
2024-04-05,Widget B,Electronics,East,90,1800,900
2024-04-10,Gadget Z,Outdoor,West,70,2100,1050
2024-04-15,Widget A,Electronics,North,155,3100,1550
2024-04-20,Gadget Y,Home,South,190,3800,1520
2024-04-25,Widget B,Electronics,West,125,2500,1250"""

df = pd.read_csv(StringIO(CSV_DATA))

def ask_gemini(question, data_summary):
    prompt = f"""You are a data analysis expert. Here is a sales dataset summary:
{data_summary}

The user asks: {question}

Instructions:
- Answer based on the data
- If they want a chart, respond with ONLY this JSON format: {{"chart": true, "type": "bar", "x": "category", "y": "revenue", "agg": "sum", "title": "Revenue by Category"}}
- chart type can be: bar, line, pie
- x and y must be actual column names from: date, product, category, region, units_sold, revenue, cost
- Otherwise just answer in plain text
- Be concise and insightful"""

    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=body, timeout=30
        )
        full_response = res.json()
        return full_response["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error: {full_response if 'full_response' in locals() else e}"

def make_chart(chart_type, x_col, y_col, agg, title):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        agg_map = {"sum": "sum", "mean": "mean", "count": "count"}
        if chart_type == "pie":
            data = df.groupby(x_col)[y_col].sum()
            ax.pie(data.values, labels=data.index, autopct="%1.1f%%")
        elif chart_type == "line":
            data = df.groupby(x_col)[y_col].agg(agg_map.get(agg, "sum"))
            ax.plot(data.index.astype(str), data.values, marker="o", color="#6366f1")
            plt.xticks(rotation=45, ha="right")
        else:
            data = df.groupby(x_col)[y_col].agg(agg_map.get(agg, "sum"))
            ax.bar(data.index.astype(str), data.values, color="#6366f1")
            plt.xticks(rotation=45, ha="right")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("/tmp/chart.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        plt.close("all")
        return False

data_summary = f"""
Shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns: {df.columns.tolist()}
Products: {df['product'].unique().tolist()}
Categories: {df['category'].unique().tolist()}
Regions: {df['region'].unique().tolist()}
Total Revenue: {df['revenue'].sum()}
Revenue by product: {df.groupby('product')['revenue'].sum().to_dict()}
Revenue by category: {df.groupby('category')['revenue'].sum().to_dict()}
Revenue by region: {df.groupby('region')['revenue'].sum().to_dict()}
Avg units sold: {df['units_sold'].mean():.1f}
"""

with st.sidebar:
    st.success(f"✅ Dataset loaded! {len(df)} rows × {len(df.columns)} cols")
    st.write("**Columns:**")
    for col in df.columns:
        st.code(col, language=None)
    st.divider()
    st.header("💡 Try These")
    samples = [
        "Give me an overview of this dataset",
        "Which product has highest revenue?",
        "Bar chart of revenue by category",
        "Which region performs best?",
        "Show a pie chart of revenue by product",
        "What is the average units sold?",
    ]
    for q in samples:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    if st.button("🗑 Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("chart"):
            st.image("/tmp/chart.png")

pending = st.session_state.pop("pending", None)
user_input = st.chat_input("Ask anything about your data...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_gemini(user_input, data_summary)

        chart_generated = False
        try:
            clean = response.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            if isinstance(data, dict) and data.get("chart"):
                chart_generated = make_chart(
                    data.get("type", "bar"),
                    data.get("x"), data.get("y"),
                    data.get("agg", "sum"),
                    data.get("title", "Chart")
                )
                answer = f"📊 Here's your {data.get('title', 'chart')}!"
            else:
                answer = response
        except:
            answer = response

        st.write(answer)
        if chart_generated:
            st.image("/tmp/chart.png")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chart": chart_generated,
    })
