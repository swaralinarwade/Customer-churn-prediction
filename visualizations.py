"""
Visualizations Module for STREAMSCOPE
Generates cinematic, restrained, editorial Plotly charts designed for the deep charcoal
and muted crimson streaming subscriber intelligence platform.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Editorial Cinematic Color Tokens
BG_DARK = "#0d0e12"           # Deep charcoal / near black
SURFACE_GRAPHITE = "#181a20"  # Slightly lighter graphite
BORDER_MUTED = "#262933"      # Subtle divider border
CRIMSON_ACCENT = "#d9232e"    # Refined muted crimson (Streaming brand accent)
WHITE_SOFT = "#f3f4f6"        # Soft white heading
TEXT_GRAY = "#8a8f98"         # Neutral supporting text
COLOR_CHURN = "#ef4444"       # Restrained coral red
COLOR_RETAIN = "#10b981"      # Muted emerald green
COLOR_AMBER = "#f59e0b"       # Restrained amber risk

def apply_editorial_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Applies executive streaming editorial styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title.upper()}</b>" if title else "",
            "y": 0.95,
            "x": 0.04,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 13, "color": WHITE_SOFT, "family": "Inter, sans-serif"}
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=55 if title else 25, b=40),
        height=height,
        font=dict(color=TEXT_GRAY, family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=WHITE_SOFT, size=11)
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=BORDER_MUTED,
        gridwidth=0.5,
        tickfont=dict(color=TEXT_GRAY, size=10),
        title_font=dict(color=TEXT_GRAY, size=11)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=BORDER_MUTED,
        gridwidth=0.5,
        tickfont=dict(color=TEXT_GRAY, size=10),
        title_font=dict(color=TEXT_GRAY, size=11)
    )
    return fig

def plot_churn_donut(df: pd.DataFrame) -> go.Figure:
    """Minimalist, editorial donut chart showing retained vs churned subscriber proportions."""
    churn_counts = df["Churn"].value_counts()
    stay_count = churn_counts.get(0, 0)
    churn_count = churn_counts.get(1, 0)

    fig = go.Figure(data=[go.Pie(
        labels=["Retained Subscribers", "Churned Accounts"],
        values=[stay_count, churn_count],
        hole=0.68,
        marker=dict(colors=[COLOR_RETAIN, CRIMSON_ACCENT], line=dict(color=BG_DARK, width=2.5)),
        textinfo="percent",
        textfont=dict(color=WHITE_SOFT, size=13),
        hoverinfo="label+value+percent"
    )])

    fig = apply_editorial_theme(fig, "Subscriber Base Distribution", height=320)
    fig.update_layout(
        showlegend=True,
        annotations=[{
            "text": f"<b>{len(df):,}</b><br><span style='font-size:10px;color:#8a8f98;'>ACCOUNTS</span>",
            "x": 0.5, "y": 0.5, "font_size": 18, "font_color": WHITE_SOFT, "showarrow": False
        }]
    )
    return fig

def plot_viewing_hours_density(df: pd.DataFrame) -> go.Figure:
    """Overlaid distribution of weekly viewing hours for churned vs retained subscribers."""
    df_clean = df.dropna(subset=["AverageWeeklyHours"])
    retained_hours = df_clean[df_clean["Churn"] == 0]["AverageWeeklyHours"]
    churned_hours = df_clean[df_clean["Churn"] == 1]["AverageWeeklyHours"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=retained_hours,
        name="Retained Subscribers",
        opacity=0.75,
        marker_color=COLOR_RETAIN,
        nbinsx=35
    ))
    fig.add_trace(go.Histogram(
        x=churned_hours,
        name="Churned Accounts",
        opacity=0.75,
        marker_color=CRIMSON_ACCENT,
        nbinsx=35
    ))

    fig = apply_editorial_theme(fig, "Weekly Viewing Hours Distribution (Engagement)", height=350)
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Average Hours Watched Per Week",
        yaxis_title="Subscriber Volume"
    )
    return fig

def plot_churn_by_plan_tier(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing churn and retention volumes across subscription tiers."""
    grouped = df.groupby(["SubscriptionPlan", "Churn"]).size().unstack(fill_value=0)
    plans = ["Basic", "Standard", "Premium"]
    # Reindex to logical tier hierarchy
    grouped = grouped.reindex(plans, fill_value=0)

    fig = go.Figure()
    if 0 in grouped.columns:
        fig.add_trace(go.Bar(
            name="Retained",
            x=grouped.index,
            y=grouped[0],
            marker_color=COLOR_RETAIN
        ))
    if 1 in grouped.columns:
        fig.add_trace(go.Bar(
            name="Churned",
            x=grouped.index,
            y=grouped[1],
            marker_color=CRIMSON_ACCENT
        ))

    fig = apply_editorial_theme(fig, "Subscriber Volumes across Plan Tiers", height=340)
    fig.update_layout(barmode="group", xaxis_title="Subscription Tier", yaxis_title="Subscribers")
    return fig

def plot_inactivity_hazard_curve(df: pd.DataFrame) -> go.Figure:
    """Plots empirical churn probability as a function of days since last active."""
    # Bin inactivity into discrete buckets
    bins = [0, 3, 7, 14, 21, 30, 45]
    labels = ["0-3 Days", "4-7 Days", "8-14 Days", "15-21 Days", "22-30 Days", "31+ Days"]
    df_temp = df.copy()
    df_temp["InactivityCohort"] = pd.cut(df_temp["DaysSinceLastActive"], bins=bins, labels=labels, include_lowest=True)

    rates = df_temp.groupby("InactivityCohort", observed=False)["Churn"].mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rates.index.astype(str),
        y=rates.values,
        mode="lines+markers",
        line=dict(color=CRIMSON_ACCENT, width=3),
        marker=dict(size=8, color=WHITE_SOFT, line=dict(color=CRIMSON_ACCENT, width=2)),
        text=[f"{v:.1f}%" for v in rates.values],
        textposition="top center"
    ))

    fig = apply_editorial_theme(fig, "Inactivity Hazard Curve: Churn Rate vs. Inactive Days", height=340)
    fig.update_layout(
        xaxis_title="Days Since Last Active Session",
        yaxis_title="Observed Churn Rate (%)",
        yaxis_range=[0, 100]
    )
    return fig

def plot_device_engagement_breakdown(df: pd.DataFrame) -> go.Figure:
    """Plots device usage distribution and respective churn rates."""
    device_stats = df.groupby("PrimaryDevice").agg(
        Total=("Churn", "count"),
        ChurnRate=("Churn", "mean")
    ).reset_index()
    device_stats["ChurnRate"] = device_stats["ChurnRate"] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=device_stats["PrimaryDevice"],
        y=device_stats["ChurnRate"],
        marker=dict(
            color=device_stats["ChurnRate"],
            colorscale=[[0, "#4b5563"], [1, CRIMSON_ACCENT]]
        ),
        text=device_stats["ChurnRate"].apply(lambda v: f"{v:.1f}%"),
        textposition="auto"
    ))

    fig = apply_editorial_theme(fig, "Churn Propensity by Primary Playback Device", height=340)
    fig.update_layout(xaxis_title="Playback Device", yaxis_title="Churn Rate (%)", yaxis_range=[0, 60])
    return fig

def plot_algorithm_benchmark(comp_df: pd.DataFrame) -> go.Figure:
    """Multi-metric comparison chart for Logistic Regression, Decision Tree, and Random Forest."""
    metrics = ["Accuracy", "Precision (Churn)", "Recall (Churn)", "F1-Score", "ROC-AUC"]
    colors = ["#475569", "#0284c7", CRIMSON_ACCENT, "#8b5cf6", COLOR_RETAIN]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        if metric in comp_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=comp_df["Algorithm"],
                y=comp_df[metric],
                marker_color=color,
                text=comp_df[metric].apply(lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)),
                textposition="auto"
            ))

    fig = apply_editorial_theme(fig, "Algorithm Evaluation Benchmarking", height=400)
    fig.update_layout(
        barmode="group",
        yaxis_title="Score (0.00 to 1.00)",
        yaxis_range=[0, 1.05]
    )
    return fig

def plot_confusion_matrix_annotated(cm_dict: dict, model_name: str) -> go.Figure:
    """Editorial heatmap for the classification confusion matrix."""
    matrix = np.array(cm_dict["matrix"])
    labels = ["Retained (0)", "Churned (1)"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    
    text = [
        [f"True Negative<br><b>{tn}</b>", f"False Positive<br><b>{fp}</b>"],
        [f"False Negative<br><b>{fn}</b>", f"True Positive<br><b>{tp}</b>"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 13, "color": WHITE_SOFT},
        colorscale=[
            [0.0, SURFACE_GRAPHITE],
            [0.5, "#374151"],
            [1.0, CRIMSON_ACCENT]
        ],
        showscale=False
    ))

    fig = apply_editorial_theme(fig, f"Confusion Matrix: {model_name}", height=360)
    fig.update_layout(
        xaxis_title="Model Predicted Outcome",
        yaxis_title="Actual Ground Truth",
        yaxis_autorange="reversed"
    )
    return fig

def plot_roc_curves_multi(eval_results: list) -> go.Figure:
    """Overlaid ROC curves with diagonal random guessing baseline."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(color="#475569", width=1.5, dash="dash"),
        name="Random Baseline (AUC = 0.50)"
    ))

    colors = [CRIMSON_ACCENT, "#8b5cf6", "#0284c7"]
    for i, res in enumerate(eval_results):
        if res.get("roc_data") and res.get("roc_auc") != "N/A":
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=res["roc_data"]["fpr"],
                y=res["roc_data"]["tpr"],
                mode="lines",
                name=f"{res['model_name']} (AUC: {res['roc_auc']:.3f})",
                line=dict(color=color, width=2.5)
            ))

    fig = apply_editorial_theme(fig, "Receiver Operating Characteristic (ROC)", height=400)
    fig.update_layout(
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall / Sensitivity)",
        xaxis_range=[0, 1.0],
        yaxis_range=[0, 1.02]
    )
    return fig

def plot_feature_importance_horizontal(feature_names: list, importances: list, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart showing top influential viewer features."""
    def clean_label(name: str) -> str:
        name = name.replace("num__", "").replace("cat__", "")
        mapping = {
            "DaysSinceLastActive": "Days Since Last Active Session",
            "AverageWeeklyHours": "Average Weekly Viewing Hours",
            "TenureMonths": "Subscriber Tenure (Months)",
            "NumberOfProfiles": "Profiles on Account",
            "MonthlyCharges": "Monthly Subscription Fee ($)",
            "CustomerSupportTickets": "Support Tickets Past 6M",
            "ContentDownloadsPerMonth": "Offline Downloads Per Month",
            "SubscriptionPlan_Premium": "Plan: 4K Premium Tier",
            "SubscriptionPlan_Standard": "Plan: Standard Tier",
            "PrimaryDevice_Smart TV": "Device: Smart TV",
            "PrimaryDevice_Mobile": "Device: Mobile Phone",
            "PaymentMethod_Direct Bank Transfer": "Payment: Direct Bank Transfer"
        }
        return mapping.get(name, name.replace("_", " "))

    cleaned_names = [clean_label(fn) for fn in feature_names]
    df_imp = pd.DataFrame({"Feature": cleaned_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation="h",
        marker=dict(
            color=df_imp["Importance"],
            colorscale=[[0, "#374151"], [1, CRIMSON_ACCENT]]
        ),
        text=df_imp["Importance"].apply(lambda v: f"{v*100:.1f}%"),
        textposition="outside"
    ))

    fig = apply_editorial_theme(fig, f"Top {top_n} Predictive Viewer Features (Random Forest)", height=400)
    fig.update_layout(xaxis_title="Relative Feature Importance Weight", yaxis_title="")
    return fig

def plot_risk_gauge_editorial(probability: float) -> go.Figure:
    """Cinematic semi-circular probability risk gauge."""
    pct = round(probability * 100, 1)

    if pct < 30:
        bar_color = COLOR_RETAIN
        risk_text = "LOW RETENTION RISK"
    elif pct < 60:
        bar_color = COLOR_AMBER
        risk_text = "MODERATE ATTRITION HAZARD"
    else:
        bar_color = CRIMSON_ACCENT
        risk_text = "CRITICAL CHURN PROPENSITY"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": WHITE_SOFT, "family": "Inter, sans-serif"}},
        title={"text": f"<b>{risk_text}</b>", "font": {"size": 13, "color": bar_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT_GRAY, "tickfont": {"color": TEXT_GRAY, "size": 10}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": SURFACE_GRAPHITE,
            "borderwidth": 1,
            "bordercolor": BORDER_MUTED,
            "steps": [
                {"range": [0, 30], "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [30, 60], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [60, 100], "color": "rgba(217, 35, 46, 0.2)"}
            ],
            "threshold": {
                "line": {"color": WHITE_SOFT, "width": 3},
                "thickness": 0.75,
                "value": pct
            }
        }
    ))

    fig = apply_editorial_theme(fig, "", height=270)
    fig.update_layout(margin=dict(l=20, r=20, t=25, b=10))
    return fig
