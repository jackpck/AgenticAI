import pandas as pd
import json

def convert_json_to_df(transcript_json_str):
    transcript_json = json.loads(transcript_json_str)
    df = pd.DataFrame(transcript_json["sections"])
    return df

def load_transcript_json(output_folder_path,
                         ticker,
                         quarter,
                         year):
    output_path = (f"{output_folder_path.rstrip('/')}"
                   f"/{ticker}_Q{quarter}_{year}_preprocessed.json")
    with open(output_path, "r", encoding="utf-8") as f:
        transcript_json_str = f.read()
    return transcript_json_str

if __name__ == "__main__":
    output_folder_path = "../data/processed"
    ticker = "NVDA"
    ticker = ticker.lower()
    quarter = 1
    year = 2026

    transcript_json_str = load_transcript_json(output_folder_path=output_folder_path,
                                               ticker=ticker,
                                               quarter=quarter,
                                               year=year)
    print(convert_json_to_df(transcript_json_str))

