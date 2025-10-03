import pandas as pd
from utils.reference_parser import parse_references

class Demo:
    def __init__(self):
        pass

    @staticmethod
    def generate_demo(question, provider, answer):
        demo_path = "data/demo.csv"

        try:
            demo_df = pd.read_csv(demo_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            demo_df = pd.DataFrame()

        references_for_csv = parse_references(answer)
        
        new_row = pd.DataFrame([{
            "question": question,
            "provider": provider,
            "answer": answer,
            "references": references_for_csv
        }])

        demo_df = pd.concat([demo_df, new_row], ignore_index=True)

        demo_df.to_csv(demo_path, index=False)
    