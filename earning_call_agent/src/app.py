from agent import EarningCallAgent
import utils
from dotenv import load_dotenv
import os
from urllib.error import URLError
import streamlit as st
import plotly.express as px
from langchain.chat_models import init_chat_model

import sys
sys.path.append("../")
from system_prompts import prompts

load_dotenv("../../venv")

API_PATH = "../../config/google_ai_studio_api.txt"
if not os.environ.get("GOOGLE_API_KEY"):
    with open(API_PATH, "r", encoding='utf-8') as f:
        api_key = f.read()
    os.environ["GOOGLE_API_KEY"] = api_key

config = {"configurable": {"thread_id": "1"}}
model = "gemini-2.5-flash"
model_provider = "google_genai"
system_preprocess_prompt = prompts.SYSTEM_PREPROCESS_PROMPT
system_analysis_prompt = prompts.SYSTEM_ANALYSIS_PROMPT
system_chatbot_prompt = prompts.SYSTEM_CHATBOT_PROMPT
TRANSCRIPT_FOLDER_PATH = "../data/raw"
OUTPUT_FOLDER_PATH = "../data/processed"
agent = EarningCallAgent(model=model,
                         model_provider=model_provider,
                         system_prompt=prompts)


try:
    ticker_list = ["NVDA","AAPL","MSFT"]
    year_list = [2025, 2026]
    quarter_list = [1,2,3,4]

    col1, col2, col3 = st.columns(3)

    with col1:
        stock = st.selectbox(
            "Choose stock", ticker_list
        ).lower()
    with col2:
        year = st.selectbox(
            "Choose year", year_list
        )
    with col3:
        quarter = st.selectbox(
            "Choose quarter", quarter_list
        )

    if not (stock or year or quarter):
        st.error("Please select a stock, year and quarter.")
    else:
        context = {"ticker": stock,
                   "year": year,
                   "quarter": quarter,
                   "transcript_folder_path": TRANSCRIPT_FOLDER_PATH,
                   "output_folder_path": OUTPUT_FOLDER_PATH}
        # to ensure the output file exists
        agent.graph.invoke(context, config)

        transcript_json_str = utils.load_transcript_json(output_folder_path=OUTPUT_FOLDER_PATH,
                                                         ticker=stock,
                                                         quarter=quarter,
                                                         year=year)
        df = utils.convert_json_to_df(transcript_json_str)

        # 1. show insights
        col_sentiment, col_risk = st.columns(2)
        with col_sentiment:
            st.subheader("Sentiment")
            df_sentiment_count = df["sentiment"].value_counts()
            fig_sentiment = px.pie(
                names = df_sentiment_count.index,
                values = df_sentiment_count.values,
            )
            st.plotly_chart(fig_sentiment)
        with col_risk:
            st.subheader("Risk")
            df_risk_count = df["risk factor"].value_counts()
            fig_risk = px.pie(
                names = df_risk_count.index,
                values = df_risk_count.values,
            )
            st.plotly_chart(fig_risk)

        # 2. chatbot
        st.subheader("Earning call transcript chatbot")

        col_sentiment_filter, col_risk_filter = st.columns(2)
        with col_sentiment_filter:
            sentiment_filter = st.multiselect(
                "Choose sentiment",
                ["positive","negative","mixed"],
                ["positive", "negative", "mixed"]
            )
            if not sentiment_filter:
                st.error("Please select at least one sentiment.")
        with col_risk_filter:
            risk_filter = st.multiselect(
                "Choose risk",
                ["yes","no"],
                ["yes", "no"]
            )
            if not risk_filter:
                st.error("Please select at least one risk.")

        filter_dict = {"sentiment": sentiment_filter,
                       "risk factor": risk_filter}

        transcript_json_filtered = utils.filter_json(transcript_json_str,
                                                     filter_dict=filter_dict)

        chatbot = init_chat_model(model=model,
                                  model_provider=model_provider)


        if "model" not in st.session_state:
            st.session_state["model"] = model

        if "model_provider" not in st.session_state:
            st.session_state["model_provider"] = model_provider

        # initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # display chat messages from history and rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("AMA about the selected earning call transcript!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                messages = system_chatbot_prompt.format(transcript_json_filtered,
                                                        st.session_state.messages[-1]["content"])
                stream = chatbot.astream(messages)
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

        if st.button("Reset chat"):
            st.session_state.clear()


        # 3. show transcript df
        st.subheader("Structured transcript")
        st.dataframe(df)


except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}")