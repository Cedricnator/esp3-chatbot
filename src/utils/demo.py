import pandas as pd

class Demo:
    def __init__(self):
        pass

    @staticmethod
    def generate_demo(question, provider, answer, references):
        demo_path = "data/demo.csv"

        try:
            demo_df = pd.read_csv(demo_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            demo_df = pd.DataFrame()

        demo_df = demo_df.append({
            "question": question,
            "provider": provider,
            "answer": answer,
            "references": references
        }, ignore_index=True)

        demo_df.to_csv(demo_path, index=False)
    