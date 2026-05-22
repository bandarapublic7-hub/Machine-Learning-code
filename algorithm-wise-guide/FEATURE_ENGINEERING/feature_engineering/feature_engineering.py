import pandas as pd


def build_booking_frame():
    return pd.DataFrame(
        {
            "full_name": [
                "Sharma, Mr. Arjun",
                "Kapoor, Mrs. Neha",
                "Khan, Miss. Sara",
            ],
            "ticket": ["PC 17599", "STON/O2 3101282", "A/5 21171"],
            "family_members": [1, 3, 0],
            "route": ["Delhi -> Goa", "Pune -> Mumbai", "Jaipur -> Delhi"],
            "booking_time": pd.to_datetime(
                ["2026-01-03 10:05", "2026-01-08 18:30", "2026-01-11 08:45"]
            ),
        }
    )


def create_features(df):
    engineered = df.copy()
    engineered["title"] = engineered["full_name"].str.extract(r",\s*([^.]*)\.")
    engineered["family_size"] = engineered["family_members"] + 1
    engineered["origin"] = engineered["route"].str.split("->").str[0].str.strip()
    engineered["destination"] = engineered["route"].str.split("->").str[1].str.strip()
    engineered["ticket_prefix"] = engineered["ticket"].str.extract(r"([A-Za-z/]+)")
    engineered["booking_hour"] = engineered["booking_time"].dt.hour
    engineered["is_weekend_booking"] = (
        engineered["booking_time"].dt.dayofweek >= 5
    ).astype(int)
    return engineered


if __name__ == "__main__":
    booking_df = build_booking_frame()
    print("Original data:")
    print(booking_df)
    print()

    print("Engineered data:")
    print(create_features(booking_df))

