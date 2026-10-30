"""
Subscriber Insights and Retention Analytics for STREAMSCOPE
Computes data-driven behavioral observations, risk segments, and strategic retention playbooks.
"""

import pandas as pd
import numpy as np

def generate_data_observations(df: pd.DataFrame) -> list:
    """
    Computes data-driven observations from the actual streaming subscriber dataset.
    Never invents statistics; calculates exact values.
    """
    total = len(df)
    churn_rate = df["Churn"].mean() * 100

    # Inactivity hazard calculation
    inactivity_split = df.groupby(df["DaysSinceLastActive"] >= 14)["Churn"].mean() * 100
    churn_inactive = inactivity_split.get(True, 0.0)
    churn_active = inactivity_split.get(False, 0.0)

    # Viewing hours under-consumption hazard
    viewing_split = df.groupby(df["AverageWeeklyHours"] < 8.0)["Churn"].mean() * 100
    churn_low_view = viewing_split.get(True, 0.0)
    churn_high_view = viewing_split.get(False, 0.0)

    # Plan tier churn rates
    plan_churn = df.groupby("SubscriptionPlan")["Churn"].mean() * 100
    basic_churn = plan_churn.get("Basic", 0.0)
    premium_churn = plan_churn.get("Premium", 0.0)

    # Tenure cohorts
    tenure_split = df.groupby(df["TenureMonths"] <= 6)["Churn"].mean() * 100
    churn_early = tenure_split.get(True, 0.0)
    churn_mature = tenure_split.get(False, 0.0)

    # Profiles impact
    profile_split = df.groupby(df["NumberOfProfiles"] == 1)["Churn"].mean() * 100
    churn_single = profile_split.get(True, 0.0)
    churn_multi = profile_split.get(False, 0.0)

    observations = [
        f"Inactivity is the single strongest behavioral churn trigger: Subscribers inactive for 14 or more days exhibit a {churn_inactive:.1f}% churn rate, compared to only {churn_active:.1f}% for active weekly streamers.",
        f"Weekly watch threshold: Viewers watching under 8 hours per week experience an attrition rate of {churn_low_view:.1f}%, versus {churn_high_view:.1f}% among engaged streamers watching 8+ hours.",
        f"Early tenure drop-off: Subscribers in their first 6 months exhibit a {churn_early:.1f}% churn rate, whereas accounts active past 6 months stabilize with a {churn_mature:.1f}% cancellation rate.",
        f"Household lock-in: Single-profile accounts churn at {churn_single:.1f}%, while multi-profile family/household accounts drop to {churn_multi:.1f}% due to shared utility and higher switching costs.",
        f"Plan tier sensitivity: Basic tier subscribers show an average cancellation rate of {basic_churn:.1f}%, compared to {premium_churn:.1f}% for 4K Premium subscribers."
    ]

    return observations

def get_retention_playbook() -> pd.DataFrame:
    """
    Returns strategic subscriber retention intervention matrix.
    """
    playbook = [
        {
            "Risk Profile": "Dormant Viewer (14+ Days Inactive)",
            "Observed Pattern": "Low recent app open, zero playback sessions",
            "Intervention Mechanism": "Personalized 'New Season / Top 10 for You' notification & email teaser",
            "Target Window": "Day 10 to Day 14 of inactivity",
            "Business Objective": "Trigger immediate re-engagement before subscription renewal cycle"
        },
        {
            "Risk Profile": "Low Engagement Streamer (<8 hrs/wk)",
            "Observed Pattern": "Under-utilizing plan value; perceives subscription as unneeded expense",
            "Intervention Mechanism": "Curated genre recommendation carousels and 'Quick 20-Min Watch' collections",
            "Target Window": "Weekly digest",
            "Business Objective": "Increase daily session frequency and habit formation"
        },
        {
            "Risk Profile": "New Subscriber (Months 1 to 3)",
            "Observed Pattern": "Steepest drop-off hazard following initial sign-up curiosity",
            "Intervention Mechanism": "Interactive onboarding tour, prompt profile setup, and download demo",
            "Target Window": "First 14 days",
            "Business Objective": "Solidify multi-device app presence and offline consumption"
        },
        {
            "Risk Profile": "Single Profile Basic Plan",
            "Observed Pattern": "Lacks family / household lock-in; higher price sensitivity",
            "Intervention Mechanism": "Free 30-day promotional upgrade to Standard (2 screens) for household sharing",
            "Target Window": "Month 2 renewal",
            "Business Objective": "Establish multi-user household dependence to increase switching cost"
        },
        {
            "Risk Profile": "Technical Friction (2+ Support Tickets)",
            "Observed Pattern": "Multiple playback, buffering, or billing dispute tickets",
            "Intervention Mechanism": "Priority concierge customer service review + 1-month $5 courtesy billing credit",
            "Target Window": "Within 24h of second ticket",
            "Business Objective": "Mitigate brand frustration and neutralize negative sentiment"
        }
    ]
    return pd.DataFrame(playbook)
