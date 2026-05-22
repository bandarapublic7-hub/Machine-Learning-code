import pandas as pd


def build_sample_data():
    return pd.DataFrame(
        {
            "age": [22, 24, 29, 31, 35, 41, 22, 29],
            "city": ["Delhi", "Pune", "Delhi", "Jaipur", "Pune", "Delhi", "Jaipur", "Pune"],
            "monthly_spend": [1200, 1600, 2100, 2300, 1900, 2700, 1250, 2050],
            "subscribed": [0, 1, 1, 1, 0, 1, 0, 1],
        }
    )


def summarize_dataset(df):
    print("Shape:", df.shape)
    print()
    print("Describe:")
    print(df.describe(include="all").transpose())
    print()


def univariate_analysis(df):
    print("City counts:")
    print(df["city"].value_counts())
    print()

    print("Age distribution summary:")
    print(df["age"].describe())
    print()


def bivariate_analysis(df):
    print("Average spend by city:")
    print(df.groupby("city")["monthly_spend"].mean())
    print()

    print("Subscription rate by city:")
    print(pd.crosstab(df["city"], df["subscribed"], normalize="index"))
    print()

    print("Numeric correlations:")
    print(df.corr(numeric_only=True))
    print()


if __name__ == "__main__":
    dataset = build_sample_data()
    summarize_dataset(dataset)
    univariate_analysis(dataset)
    bivariate_analysis(dataset)

