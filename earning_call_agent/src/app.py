from src.agent import EarningCallAgent


config = {"configurable": {"thread_id": "1"}}
model = "gemini-2.5-flash"
model_provider = "google_genai"
TRANSCRIPT_FOLDER_PATH = "../data/raw"
OUTPUT_FOLDER_PATH = "../data/processed"
agent = EarningCallAgent(model=model,
                         model_provider=model_provider)

try:
    ticker_list = ["NVDA","AAPL"]
    stocks = st.selectbox(
        "Choose stock", ticker_list
    )
    if not stocks:
        st.error("Please select at least one stock.")
    else:
        context = {"ticker": stocks,
                   "year": 2026,
                   "quarter": 1,
                   "transcript_folder_path":TRANSCRIPT_FOLDER_PATH,
                   "output_folder_path":OUTPUT_FOLDER_PATH}
        agent.invoke(context, config)

        st.subheader("Gross agricultural production ($B)")
        st.dataframe(agent["transcript_json"])

        ########################################################
        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
        )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                color="Region:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)
except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}"):memoryview