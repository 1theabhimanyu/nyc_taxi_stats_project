import pandas as pd
import scipy.stats as stats

def load_and_clean_data(data_url: str) -> pd.DataFrame:
    print("[INFO] Fetching and cleaning dataset for statistical testing...")
    df = pd.read_csv(data_url).drop_duplicates()
    
    df = df.dropna(subset=["pickup_zone", "dropoff_zone", "pickup_borough", "dropoff_borough"]).copy()
    
    df.loc[df["payment"].isnull() & (df["tip"] > 0), "payment"] = "credit card"
    df["payment"] = df["payment"].fillna(df["payment"].mode()[0])

    df["pickup"] = pd.to_datetime(df["pickup"])
    df["dropoff"] = pd.to_datetime(df["dropoff"])
    df["trip_duration_mins"] = (df["dropoff"] - df["pickup"]).dt.total_seconds() / 60.0

    valid_mask = (
        (df["distance"] > 0.1) & 
        (df["trip_duration_mins"] > 0.5) & 
        (df["trip_duration_mins"] <= 180.0) & 
        (df["passengers"] > 0) & 
        (df["fare"] > 0)
    )
    df_clean = df[valid_mask].copy()
    return df_clean

def check_assumptions(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PRELIMINARY ASSUMPTION CHECKS")
    print("=" * 60)
    
    k2_stat, k2_p = stats.normaltest(df["fare"])
    print(f"D'Agostino K^2 Normality Test (Fare): stat={k2_stat:.2f}, p={k2_p:.4e}")
    
    qns_fare = df.loc[df["pickup_borough"] == "Queens", "fare"]
    man_fare = df.loc[df["pickup_borough"] == "Manhattan", "fare"]
    lev_stat, lev_p = stats.levene(qns_fare, man_fare)
    print(f"Levene's Test for Equality of Variance (Queens vs Manhattan Fare): stat={lev_stat:.2f}, p={lev_p:.4e}")