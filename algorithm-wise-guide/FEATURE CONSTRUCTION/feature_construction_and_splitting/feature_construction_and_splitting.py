import pandas as pd


df = pd.DataFrame(
    {
        "route": ["Delhi -> Goa", "Pune -> Mumbai", "Jaipur -> Delhi"],
        "family_members": [1, 3, 0],
        "full_name": ["Sharma, Mr. Arjun", "Kapoor, Mrs. Neha", "Khan, Miss. Sara"],
    }
)

df["origin"] = df["route"].str.split("->").str[0].str.strip()
df["destination"] = df["route"].str.split("->").str[1].str.strip()
df["family_size"] = df["family_members"] + 1
df["title"] = df["full_name"].str.extract(r",\s*([^.]*)\.")

print(df)
