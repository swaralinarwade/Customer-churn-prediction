"""
Dataset Generation and Verification Module for STREAMSCOPE
Creates a realistic, statistically grounded educational dataset of streaming subscribers,
modeling viewing hours, inactivity, subscription tiers, device usage, and churn hazards.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "streaming_churn.csv")

def generate_streaming_dataset(n_samples: int = 6000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic educational streaming subscriber dataset.
    Features follow empirical distributions observed in subscription video-on-demand (SVOD) platforms.
    """
    np.random.seed(random_state)

    # 1. Subscriber IDs
    customer_ids = [f"STR-{10000 + i}" for i in range(n_samples)]

    # 2. Subscription Plans
    # Industry distribution: Basic (~28%), Standard (~45%), Premium (~27%)
    plans = np.random.choice(["Basic", "Standard", "Premium"], size=n_samples, p=[0.28, 0.45, 0.27])

    # 3. Monthly Fee mapped with small realistic regional/promotional variance
    base_fee_map = {"Basic": 9.99, "Standard": 15.49, "Premium": 19.99}
    monthly_fees = [
        round(base_fee_map[p] + np.random.choice([-1.0, 0.0, 0.5, 1.0], p=[0.1, 0.7, 0.1, 0.1]), 2)
        for p in plans
    ]

    # 4. Subscriber Tenure (Months) - Exponential distribution modeling subscriber hazard
    tenure_raw = np.random.exponential(scale=18.0, size=n_samples)
    tenure = np.clip(np.round(tenure_raw).astype(int), 1, 72)

    # 5. Average Weekly Viewing Hours
    # Normal distribution around 15.5 hrs, bounded between 1.0 and 48.0
    viewing_hours = np.clip(np.random.normal(loc=15.5, scale=8.2, size=n_samples), 1.0, 48.0)
    viewing_hours = np.round(viewing_hours, 1)

    # 6. Days Since Last Activity (Key engagement indicator)
    # Most active users login within 1-7 days; disengaged users stretch to 30+ days
    days_inactive_raw = np.random.exponential(scale=7.5, size=n_samples)
    days_inactive = np.clip(np.round(days_inactive_raw).astype(int), 0, 45)

    # 7. Primary Device
    # Smart TV (50%), Mobile (26%), Desktop (14%), Tablet (10%)
    devices = np.random.choice(
        ["Smart TV", "Mobile", "Desktop", "Tablet"],
        size=n_samples,
        p=[0.50, 0.26, 0.14, 0.10]
    )

    # 8. Number of Profiles on Account
    # Correlated with plan tier: Premium has more profiles
    num_profiles = []
    for p in plans:
        if p == "Basic":
            num_profiles.append(np.random.choice([1, 2], p=[0.85, 0.15]))
        elif p == "Standard":
            num_profiles.append(np.random.choice([1, 2, 3], p=[0.30, 0.55, 0.15]))
        else: # Premium
            num_profiles.append(np.random.choice([2, 3, 4, 5], p=[0.15, 0.35, 0.35, 0.15]))
    num_profiles = np.array(num_profiles)

    # 9. Payment Method
    payment_methods = np.random.choice(
        ["Credit Card", "Debit Card", "Digital Wallet", "Direct Bank Transfer"],
        size=n_samples,
        p=[0.42, 0.28, 0.20, 0.10]
    )

    # 10. Content Downloads Per Month (Offline viewing)
    downloads = np.clip(np.round(np.random.poisson(lam=4.5, size=n_samples)), 0, 25).astype(int)

    # 11. Favorite Content Genre
    genres = np.random.choice(
        ["Drama", "Sci-Fi / Fantasy", "Comedy", "Action & Thriller", "Documentary"],
        size=n_samples,
        p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    # 12. Support Tickets in Past 6 Months
    tickets = np.clip(np.random.poisson(lam=0.8, size=n_samples), 0, 6).astype(int)

    # -------------------------------------------------------------------------
    # Statistically Grounded Churn Probability Calculation (SVOD Economics)
    # -------------------------------------------------------------------------
    # Base churn baseline logit (~22% baseline in SVOD)
    logit = -1.35

    # Inactivity hazard: strongest churn indicator
    # Users with > 14 days inactivity experience steep churn risk
    logit += (days_inactive - 7.0) * 0.12

    # Under-consumption hazard: low viewing hours (< 8 hrs/week) increases churn
    logit -= (viewing_hours - 15.0) * 0.085

    # New subscriber hazard: tenure < 6 months has higher drop-off
    logit -= (np.log1p(tenure) - 2.8) * 0.45

    # Family lock-in: more profiles decreases churn
    logit -= (num_profiles - 2.0) * 0.28

    # Technical friction: frequent customer support complaints increase churn
    logit += (tickets - 0.5) * 0.35

    # Plan tier elasticity: Basic plan users churn slightly more easily
    plan_effect = np.array([0.25 if p == "Basic" else (-0.15 if p == "Premium" else 0.0) for p in plans])
    logit += plan_effect

    # Device effect: Smart TV users are more engaged than mobile-only users
    device_effect = np.array([-0.20 if d == "Smart TV" else (0.15 if d == "Mobile" else 0.0) for d in devices])
    logit += device_effect

    # Offline download stickiness
    logit -= (downloads - 3.0) * 0.04

    # Convert logit to probability via sigmoid function
    churn_prob = 1.0 / (1.0 + np.exp(-logit))

    # Realize binary churn outcomes
    churn = (np.random.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "SubscriptionPlan": plans,
        "MonthlyCharges": monthly_fees,
        "TenureMonths": tenure,
        "AverageWeeklyHours": viewing_hours,
        "DaysSinceLastActive": days_inactive,
        "PrimaryDevice": devices,
        "NumberOfProfiles": num_profiles,
        "PaymentMethod": payment_methods,
        "ContentDownloadsPerMonth": downloads,
        "FavoriteGenre": genres,
        "CustomerSupportTickets": tickets,
        "Churn": churn
    })

    # Introduce a small number of realistic missing values for preprocessing robustness
    missing_indices = np.random.choice(n_samples, size=18, replace=False)
    df.loc[missing_indices[:9], "AverageWeeklyHours"] = np.nan
    df.loc[missing_indices[9:], "ContentDownloadsPerMonth"] = np.nan

    return df

def ensure_dataset():
    """Ensure dataset exists on disk; generate if missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        print(f"[INFO] Generating STREAMSCOPE educational dataset at: {CSV_PATH}")
        df = generate_streaming_dataset(n_samples=6000, random_state=42)
        df.to_csv(CSV_PATH, index=False)
        print(f"[SUCCESS] Dataset created: {df.shape[0]} rows, {df.shape[1]} columns.")
        print(f"          Churn rate: {df['Churn'].mean()*100:.2f}% (Retained: {(1-df['Churn'].mean())*100:.2f}%)")
    else:
        df = pd.read_csv(CSV_PATH)
        print(f"[INFO] Existing dataset verified at: {CSV_PATH} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    ensure_dataset()
