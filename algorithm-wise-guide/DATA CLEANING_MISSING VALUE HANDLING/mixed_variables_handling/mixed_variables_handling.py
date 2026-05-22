import pandas as pd


df = pd.DataFrame(
    {
        "apartment": ["2 BHK", "3 BHK", "1 BHK"],
        "price_text": ["INR 35000", "INR 52000", "INR 22000"],
    }
)

df["rooms"] = df["apartment"].str.extract(r"(\d+)").astype(int)
df["price"] = df["price_text"].str.extract(r"(\d+)").astype(int)

print(df)
