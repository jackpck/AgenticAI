import pandas as pd
import json
import copy

def convert_json_to_df(transcript_json_str):
    transcript_json = json.loads(transcript_json_str)
    df = pd.DataFrame(transcript_json["sections"])
    return df

def filter_json(transcript_json_str: str,
                filter_dict: dict) -> str:
    transcript_json = json.loads(transcript_json_str)
    filtered_section = [s for s in transcript_json["sections"] if
                        all(
                            (not filter_dict[k]) or s[k] in filter_dict[k]
                            for k in filter_dict
                        )]
    filtered_transcript_json = copy.copy(transcript_json)
    filtered_transcript_json["sections"] = filtered_section
    return json.dumps(filtered_transcript_json)

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
    filter_dict = {"risk factor": "yes",
                   "sentiment": ["positive","negative"]}
    print(filter_json(transcript_json_str, filter_dict))
