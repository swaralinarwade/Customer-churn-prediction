# MINI-PROJECT PRESENTATION OUTLINE (PPT SLIDE DECK)
## Title: Netflix Customer Churn Prediction & Viewer Engagement Analytics
### Platform: STREAMSCOPE (Subscriber Intelligence Platform)
**Third-Year Computer Engineering | Machine Learning Laboratory**

---

### SLIDE 1: Title & Candidate Information
- **Slide Title:** STREAMSCOPE: Subscriber Intelligence Platform
- **Subtitle:** Streaming Customer Churn Prediction and Viewer Engagement Analytics
- **Candidate Details:** Name, Roll Number, Department of Computer Engineering
- **Academic Context:** Third-Year B.E. / B.Tech Machine Learning Laboratory Mini-Project (2025–2026)
- **Visual Design:** Minimalist dark-charcoal background with the `◬ STREAMSCOPE` wordmark in muted crimson.

---

### SLIDE 2: Problem Statement & Motivation
- **Slide Title:** The Retention Dilemma in Subscription Streaming (SVOD)
- **Key Points:**
  - **Fierce Competition & Zero Friction:** With no long-term contracts, streaming subscribers can cancel or switch platforms with a single click.
  - **High Customer Acquisition Cost (CAC):** Acquiring a new streaming user costs **5x to 7x more** than retaining an existing subscriber.
  - **The Flaw of Existing CRM:** Retention campaigns are **reactive**—offering deals only *after* account cancellation is initiated.
  - **Our Solution:** A proactive Machine Learning system that flags at-risk subscribers weeks in advance by monitoring behavioral disengagement signals.
- **Presenter Talking Point:** *"Subscribers don't cancel on a whim; they show measurable behavioral warning signs—like lengthening days of inactivity and dropping watch time—weeks before they hit the cancel button."*

---

### SLIDE 3: Project Objectives & Scope
- **Slide Title:** System Goals & Deliverables
- **Key Points:**
  - Ingest and analyze streaming subscriber telemetry (plan tier, tenure, weekly watch hours, inactivity latency, device usage).
  - Train and benchmark three classification algorithms: **Logistic Regression**, **Decision Tree**, and **Random Forest**.
  - Address the real-world **73.3% / 26.7% class imbalance** using cost-sensitive class balancing.
  - Prioritize **Recall on Churn** to minimize costly False Negatives.
  - Build a cinematic, editorial **Streamlit dashboard** (**STREAMSCOPE**) featuring horizontal telemetry strips and dynamic risk dials.
- **Visual:** Flow diagram: Raw Telemetry -> Leak-Free Preprocessing -> Multi-Model Train -> Held-Out Test Evaluation -> STREAMSCOPE UI.

---

### SLIDE 4: Streaming Subscriber Telemetry (Dataset)
- **Slide Title:** Viewer Behavior & Dataset Telemetry (N = 6,000)
- **Key Points:**
  - **Sample Size:** 6,000 subscriber accounts (4,399 Retained / 1,601 Churned).
  - **Viewing Intensity:** Average 15.5 hours/week (ranges from 0.0 to 48.0 hrs).
  - **Inactivity Latency:** Gaps between active playback sessions (0 to 45 days).
  - **Plan Tiers:** Basic ($9.99), Standard ($15.49), Premium ($19.99).
  - **Device Ecosystem:** Smart TV (50%), Mobile (26%), Desktop (14%), Tablet (10%).
  - **Household Sharing:** Profiles per account (1 to 5).
- **Visual:** Donut chart showing 73.3% retention baseline alongside an Inactivity Hazard curve.

---

### SLIDE 5: System Architecture & Data Leakage Prevention
- **Slide Title:** Modular Pipeline & Leakage-Free Architecture
- **Key Points:**
  - **Stratified Partition:** 80% Training (4,800 accounts), 20% Held-Out Testing (1,200 accounts).
  - **Scikit-Learn ColumnTransformer:**
    - Numerical Pipeline (7 features): Median Imputation + `StandardScaler`.
    - Categorical Pipeline (4 features): Mode Imputation + `OneHotEncoder(drop='first', handle_unknown='ignore')`.
  - **Data Leakage Guarantee:** Imputers and scalers are fitted **strictly on the training split**, ensuring unbiased test evaluation.
  - **Model Packaging:** Complete workflow bundled inside Scikit-Learn `Pipeline` and persisted via `joblib`.
- **Visual:** ColumnTransformer architecture diagram showing numerical and categorical sub-pipelines.

---

### SLIDE 6: Algorithms Implemented & Technical Rationale
- **Slide Title:** Supervised Classification Algorithms
- **Key Points:**
  - **1. Logistic Regression:**
    - Linear probabilistic baseline with logistic sigmoid mapping.
    - Optimized using Quasi-Newton solver (`lbfgs`) with balanced class weights.
  - **2. Decision Tree Classifier:**
    - Non-parametric recursive splitting minimizing Gini impurity.
    - Constrained depth (`max_depth=5`) to prevent overfitting.
  - **3. Random Forest Classifier:**
    - Bagged ensemble of 100 decorrelated decision trees.
    - Random subspace feature sampling ($\sqrt{p}$) reduces variance without increasing bias.
- **Visual:** Decision Tree binary split diagram contrasted with Random Forest ensemble voting.

---

### SLIDE 7: Experimental Results & Benchmarking
- **Slide Title:** Model Benchmarking on Held-Out Test Set (N = 1,200)
- **Comparative Table:**

| Algorithm | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **73.25%** | **0.4990** | **74.69%** | **0.5982** | **0.7991** |
| **Random Forest** | **74.50%** | **0.5168** | **67.19%** | **0.5842** | **0.7899** |
| **Decision Tree** | 71.25% | 0.4682 | 57.50% | 0.5161 | 0.7508 |

- **Presenter Talking Point:** *"A naive model that simply predicts 'Retained' for every user achieves 73.3% accuracy, but catches 0% of churners. By using balanced class weighting, our models achieve ~70-75% Recall—catching 3 out of every 4 at-risk subscribers."*

---

### SLIDE 8: Confusion Matrix & Cost Asymmetry Rationale
- **Slide Title:** Diagnostic Analysis: Confusion Matrix & Cost Asymmetry
- **Key Points:**
  - **Logistic Regression Test Results (N = 1,200):**
    - **True Negatives (TN):** 640 (Correctly identified loyal accounts)
    - **False Positives (FP):** 240 (Loyal accounts flagged for retention inspection)
    - **False Negatives (FN):** 81 (At-risk subscribers missed)
    - **True Positives (TP):** 239 (Churners successfully intercepted!)
  - **Cost Asymmetry in SVOD:** A False Negative costs the loss of a subscriber and $120–$240 in recurring annual revenue. A False Positive costs a fraction of a cent for an automated push notification or email recommendation.
- **Visual:** Annotated 2x2 Confusion Matrix Heatmap graphic.

---

### SLIDE 9: Feature Importance & Behavioral Drivers (XAI)
- **Slide Title:** What Drives Streaming Subscriber Churn?
- **Key Points:**
  - **Top Predictive Features in Random Forest:**
    1. **Days Since Last Active (26.4%):** Strongest indicator; 14+ days of inactivity quadruples churn risk.
    2. **Weekly Viewing Hours (21.8%):** Viewers under 8 hrs/week have double the attrition of 15+ hr streamers.
    3. **Subscriber Tenure (17.2%):** Months 1 to 6 exhibit the highest drop-off hazard before habit loops solidify.
    4. **Profiles on Account (11.5%):** Multi-profile accounts show strong household retention lock-in.
  - **Academic Distinction:** *Correlation does not equal Causation.* Features represent predictive associations, not definitive causal proof.
- **Visual:** Horizontal bar chart of top feature importance weights.

---

### SLIDE 10: STREAMSCOPE User Interface Walkthrough
- **Slide Title:** Interactive Platform Demonstration (5 Sections)
- **Key Points:**
  - **Section A: Overview Dashboard:** Live horizontal telemetry strip and real-time subscriber metrics.
  - **Section B: Subscriber Prediction Tool:** Persona presets (**High-Risk** vs. **Loyal**), real-time risk dial gauge, and explainable risk factor breakdown.
  - **Section C: Viewer Engagement:** Watch hour distributions and offline download stickiness.
  - **Section D: Model Lab:** Model comparison table, confusion matrix inspector, and ROC curves.
  - **Section E: Retention Insights:** Strategic retention intervention playbook for streaming operators.
- **Visual:** UI screenshots of the dark-charcoal STREAMSCOPE dashboard showing the risk gauge dial.

---

### SLIDE 11: Strategic Retention Playbook for Streaming Operators
- **Slide Title:** Data-Driven Retention Interventions
- **Key Points:**
  - **1. Dormant Viewer Re-Activation:** Trigger automated "New Season / Top 10 for You" teasers between Day 10 and Day 14 of inactivity.
  - **2. Under-Consumption Nudge:** Curate "Quick 20-Min Watch" and personalized genre carousels for viewers watching <8 hrs/week.
  - **3. The 90-Day Onboarding Journey:** Guide new subscribers to set up multiple profiles and test offline downloads to build multi-device habits.
  - **4. Support Ticket Follow-up:** Automatically issue a $5 courtesy credit following a second technical playback dispute.
- **Visual:** Strategic retention matrix graphic.

---

### SLIDE 12: Conclusion & Future Scope
- **Slide Title:** Summary & Future Enhancements
- **Summary:**
  - Successfully developed, tested, and validated an end-to-end Machine Learning intelligence platform for streaming subscriber retention.
  - Models achieved **74.69% Recall** on churn and **0.7991 ROC-AUC** on held-out test data.
  - Built an editorial, cinematic Streamlit interface with 8 automated unit tests (100% passing).
- **Future Scope:**
  - Implement **Survival Analysis** to forecast *exact time-to-churn in months*.
  - Integrate **SHAP Force Plots** for instance-level feature explainability.
  - Connect with automated marketing webhooks (e.g. Braze, HubSpot) for automated push notifications.
- **Thank You / Open for Viva Defense & Questions.**
