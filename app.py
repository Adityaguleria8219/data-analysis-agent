
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import json
import os
import operator
from typing import TypedDict, Annotated
from io import StringIO

GROQ_API_KEY = os.environ.get("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", "")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

st.set_page_config(page_title="Data Analysis Agent", page_icon="⚡", layout="wide")
st.title("⚡ Data Analysis Agent")
st.caption("Powered by LangGraph + Groq")

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

_df = pd.read_csv(StringIO(CSV_DATA))

def _require_df(): return _df

@tool
def describe_dataset(dummy: str = "") -> str:
    """Returns shape, columns, dtypes and basic stats. Always call this first."""
    df = _require_df()
    return json.dumps({
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {c: str(t) for c,t in df.dtypes.items()},
        "sample": df.head(3).to_dict(orient="records")
    }, indent=2)

@tool
def run_pandas_query(query_code: str) -> str:
    """Runs a pandas expression on df. Do not redefine df.
    Examples:
      df.groupby('category')['revenue'].sum().to_dict()
      df.sort_values('revenue', ascending=False).head(5).to_dict(orient='records')
    """
    df = _require_df()
    try:
        result = eval(query_code, {"df": df, "pd": pd, "json": json})
        if isinstance(result, pd.DataFrame): return result.to_json(orient="records")
        if isinstance(result, pd.Series): return result.to_json()
        return json.dumps(result, default=str)
    except Exception as e:
        return f"Error: {e}"

@tool
def generate_chart(chart_spec: str) -> str:
    """Generates a chart and saves it.
    chart_spec JSON keys:
      chart_type: bar | line | pie | scatter | hist
      x: column for x axis
      y: column for y axis
      title: chart title
      agg: sum | mean | count
    """
    df = _require_df()
    try:
        spec = json.loads(chart_spec)
        chart_type = spec.get("chart_type","bar")
        x_col = spec.get("x"); y_col = spec.get("y")
        title = spec.get("title","Chart"); agg = spec.get("agg","sum")
        fig, ax = plt.subplots(figsize=(8,5))
        if chart_type == "hist":
            df[y_col or x_col].hist(ax=ax, bins=15, color="#6366f1")
        elif chart_type == "scatter":
            ax.scatter(df[x_col], df[y_col], color="#6366f1", alpha=0.7)
        elif chart_type == "pie":
            data = df.groupby(x_col)[y_col].sum()
            ax.pie(data.values, labels=data.index, autopct="%1.1f%%")
        else:
            agg_map = {"sum":"sum","mean":"mean","count":"count"}
            data = df.groupby(x_col)[y_col].agg(agg_map.get(agg,"sum"))
            if chart_type == "line":
                ax.plot(data.index.astype(str), data.values, marker="o", color="#6366f1")
            else:
                ax.bar(data.index.astype(str), data.values, color="#6366f1")
            plt.xticks(rotation=45, ha="right")
        ax.set_title(title); plt.tight_layout()
        plt.savefig("/tmp/chart.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return json.dumps({"status":"Chart saved","title":title})
    except Exception as e:
        plt.close("all"); return json.dumps({"error":str(e)})

@tool
def get_column_insights(column_name: str) -> str:
    """Returns detailed stats for a single column."""
    df = _require_df()
    if column_name not in df.columns:
        return f"Column '{column_name}' not found. Available: {df.columns.tolist()}"
    col = df[column_name]
    result = {"column":column_name,"dtype":str(col.dtype),"nulls":int(col.isnull().sum()),"unique":int(col.nunique())}
    if pd.api.types.is_numeric_dtype(col):
        result.update({"min":float(col.min()),"max":float(col.max()),"mean":float(col.mean()),"median":float(col.median()),"std":float(col.std())})
    else:
        result["top_values"] = col.value_counts().head(10).to_dict()
    return json.dumps(result, indent=2)

ALL_TOOLS = [describe_dataset, run_pandas_query, generate_chart, get_column_insights]

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

SYSTEM_PROMPT = """You are an expert Data Analysis Agent with access to a loaded CSV dataset.
You have 4 tools:
1. describe_dataset — always call this FIRST
2. run_pandas_query — run aggregations and filters
3. generate_chart — create charts
4. get_column_insights — deep dive into one column
Always run actual queries, never guess numbers."""

@st.cache_resource
def get_agent():
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    def call_llm(state):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}
    def should_continue(state):
        last = state["messages"][-1]
        return "tools" if (hasattr(last,"tool_calls") and last.tool_calls) else END
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("tools", "llm")
    return graph.compile()

def run_agent(user_message):
    agent = get_agent()
    messages = [HumanMessage(content=user_message)]
    result = agent.invoke({"messages": messages})
    final_messages = result["messages"]
    tool_calls_made = []
    for msg in final_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc["name"])
    final_text = final_messages[-1].content
    if isinstance(final_text, list):
        final_text = " ".join(b.get("text","") for b in final_text if isinstance(b,dict))
    return {"text": final_text, "tool_calls": tool_calls_made}

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.success(f"✅ Dataset loaded! {len(_df)} rows × {len(_df.columns)} cols")
    st.write("**Columns:**")
    for col in _df.columns:
        st.code(col, language=None)
    st.divider()
    st.header("💡 Try These")
    samples = [
        "Give me an overview",
        "Which product has highest revenue?",
        "Bar chart of revenue by category",
        "Which region performs best?",
        "Insights on the revenue column",
    ]
    for q in samples:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    if st.button("🗑 Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("chart"):
            st.image("/tmp/chart.png")
        if msg.get("tool_calls"):
            st.caption(f"🔧 Tools: {', '.join(msg['tool_calls'])}")

pending = st.session_state.pop("pending", None)
user_input = st.chat_input("Ask anything about your data...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent(user_input)
        st.write(result["text"])
        chart_generated = "generate_chart" in result["tool_calls"]
        if chart_generated:
            st.image("/tmp/chart.png")
        if result["tool_calls"]:
            st.caption(f"🔧 Tools: {', '.join(result['tool_calls'])}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["text"],
        "tool_calls": result["tool_calls"],
        "chart": chart_generated,
    })
