import pandas as pd


def build_dataset():
    return pd.DataFrame(
        {
            "age": [22, 24, 29, 31, 35, None, 27],
            "city": ["Delhi", "Pune", "Delhi", "Jaipur", None, "Pune", "Delhi"],
            "monthly_spend": [1200, 1600, 2100, 2300, 1900, 1700, 1500],
            "subscribed": [0, 1, 1, 1, 0, 1, 0],
        }
    )


def create_profile_report(df, output_path="profile_report.html"):
    try:
        from ydata_profiling import ProfileReport
    except ImportError as exc:
        raise SystemExit(
            "Install the modern package first: pip install -U ydata-profiling"
        ) from exc

    profile = ProfileReport(
        df,
        title="Sample Data Profiling Report",
        explorative=True,
    )
    profile.to_file(output_path)
    return output_path


if __name__ == "__main__":
    frame = build_dataset()
    saved_path = create_profile_report(frame)
    print(f"Profile report created at: {saved_path}")

