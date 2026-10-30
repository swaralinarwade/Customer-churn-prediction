# ACADEMIC MINI-PROJECT REPORT
## NETFLIX CUSTOMER CHURN PREDICTION AND VIEWER ENGAGEMENT ANALYTICS
### System Implementation: STREAMSCOPE (Subscriber Intelligence Platform)

**Department of Computer Engineering**  
**Third-Year Computer Engineering (B.E. / B.Tech)**  
**Subject:** Machine Learning Laboratory (Mini-Project)  
**Academic Year:** 2025–2026  

---

### TABLE OF CONTENTS
1. [Abstract](#1-abstract)
2. [Introduction and Problem Statement](#2-introduction-and-problem-statement)
3. [Literature Survey and Streaming Industry Context](#3-literature-survey-and-streaming-industry-context)
4. [Dataset Description & Viewing Telemetry](#4-dataset-description--viewing-telemetry)
5. [Methodology & System Architecture](#5-methodology--system-architecture)
6. [Mathematical Formulations of Machine Learning Algorithms](#6-mathematical-formulations-of-machine-learning-algorithms)
7. [Data Preprocessing & Data Leakage Prevention](#7-data-preprocessing--data-leakage-prevention)
8. [Experimental Setup and Training Pipeline](#8-experimental-setup-and-training-pipeline)
9. [Experimental Results & Algorithmic Benchmarking](#9-experimental-results--algorithmic-benchmarking)
10. [Feature Importance & Behavioral Drivers](#10-feature-importance--behavioral-drivers)
11. [STREAMSCOPE User Interface Design System](#11-streamscope-user-interface-design-system)
12. [Strategic Retention Playbook & Causality Boundaries](#12-strategic-retention-playbook--causality-boundaries)
13. [Conclusion and Future Work](#13-conclusion-and-future-work)
14. [References](#14-references)

---

### 1. ABSTRACT
In the subscription video-on-demand (SVOD) streaming industry, subscriber defection (*churn*) directly erodes recurring monthly revenue. Customer acquisition in competitive streaming markets costs substantially more than subscriber retention. This project develops **STREAMSCOPE**, an end-to-end Machine Learning intelligence platform designed to predict streaming subscriber churn, compute individual attrition probabilities, and analyze viewer engagement telemetry.

Using an educational benchmark dataset of 6,000 streaming accounts, three classification algorithms—Logistic Regression, Decision Tree Classifier, and Random Forest Classifier—were trained using a rigorous Scikit-Learn Pipeline preventing data leakage. By implementing cost-sensitive class balancing on the 73.3% retention / 26.7% churn distribution, the models achieved a **74.69% Recall** on the churn class and an **ROC-AUC of 0.7991** on a held-out test dataset ($N = 1,200$). A cinematic, editorial web application built with Streamlit and Plotly features horizontal telemetry metric strips, dynamic risk gauges, viewer engagement distributions, and an actionable retention intervention playbook.

---

### 2. INTRODUCTION AND PROBLEM STATEMENT

#### 2.1 Background
The global streaming ecosystem (exemplified by platforms like Netflix, Disney+, and Amazon Prime Video) relies on monthly recurring revenue (MRR). With the elimination of long-term lock-in contracts, subscribers can cancel or switch providers with a single click.

#### 2.2 Problem Statement
Traditional customer retention in streaming services is almost entirely **reactive**: retention offers, surveys, or discounts are triggered only after a subscriber initiates an account cancellation flow. By that time, the user's viewing habit loop has already degraded, resulting in low win-back success rates.

Streaming platforms require an automated, **predictive decision support system** that identifies subtle pre-churn behavioral markers—such as lengthening gaps between login sessions, falling weekly watch time, and technical support disputes—several weeks prior to subscription renewal.

#### 2.3 Project Objectives
1. Implement a clean machine learning workflow analyzing viewing habits, subscription plans, device preferences, and churn outcomes.
2. Train, benchmark, and persist three classification algorithms on identical held-out test data.
3. Optimize for **Recall on Churn** to minimize costly False Negatives (undetected customer attrition).
4. Provide local and global model explainability through Random Forest feature importances.
5. Deploy a custom, cinematic web application (**STREAMSCOPE**) inspired by streaming product analytics.

---

### 3. LITERATURE SURVEY AND STREAMING INDUSTRY CONTEXT

| Study / Source | Methodology | Findings | Limitations |
| :--- | :--- | :--- | :--- |
| **Traditional CRM Analytics** | Static threshold alerts (e.g. days since login) | Simple to implement on SQL databases. | Fails to detect complex interactions between plan tier, downloads, and support disputes. |
| **Burez & Van den Poel (2009)** | Weighted Random Forests & Undersampling | Proved that handling class imbalance via cost-weighting improves churn recall. | Focused purely on telecom data rather than streaming media consumption telemetry. |
| **Gomez-Uribe & Hunt (Netflix, 2015)** | Recommendation engines & habit formation | Demonstrated that viewing engagement in the first 2 weeks strongly determines long-term subscriber retention. | Proprietary algorithmic research without open source implementation. |
| **STREAMSCOPE (Proposed)** | **Pipeline-Encapsulated Multi-Model ML with Editorial Streamlit UI** | **Balances Recall (74.7%) and ROC-AUC (0.799) with real-time persona simulation and risk dial.** | **Demonstration dataset designed for academic research.** |

---

### 4. DATASET DESCRIPTION & VIEWING TELEMETRY

The platform utilizes an educational streaming subscriber dataset ($N = 6,000$ accounts) modeling real-world SVOD behavioral characteristics:
- **Total Records:** 6,000 subscriber accounts
- **Target Variable (`Churn`):**
  - Retained (`0`): 4,399 accounts (73.32%)
  - Churned (`1`): 1,601 accounts (26.68%)

#### Feature Breakdown:
1. **Identifier:** `customerID` (unique string, excluded from model training to prevent memorization).
2. **Subscription Attributes:**
   - `SubscriptionPlan`: Basic ($9.99), Standard ($15.49), Premium ($19.99)
   - `MonthlyCharges`: Numerical fee ($7.99 to $21.99)
   - `TenureMonths`: Continuous subscriber age (1 to 72 months)
3. **Engagement Telemetry:**
   - `AverageWeeklyHours`: Weekly watch time (0.0 to 48.0 hours)
   - `DaysSinceLastActive`: Inactivity latency since last playback session (0 to 45 days)
   - `ContentDownloadsPerMonth`: Frequency of offline mobile/tablet downloads (0 to 25)
4. **Account & Hardware:**
   - `PrimaryDevice`: Smart TV (50%), Mobile (26%), Desktop (14%), Tablet (10%)
   - `NumberOfProfiles`: Profiles created on account (1 to 5)
   - `PaymentMethod`: Credit Card, Debit Card, Digital Wallet, Direct Bank Transfer
   - `FavoriteGenre`: Drama, Sci-Fi/Fantasy, Comedy, Action/Thriller, Documentary
   - `CustomerSupportTickets`: Technical or billing dispute count (0 to 6)

---

### 5. METHODOLOGY & SYSTEM ARCHITECTURE

```
+--------------------------------------------------------------+
|       STREAMSCOPE Streaming Subscriber Dataset (N=6,000)     |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|     Data Sanitization: Drop ID, Validate Numeric Types       |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|     Stratified 80/20 Train/Test Split (Preserves Churn %)    |
|             Train: 4,800  |  Held-Out Test: 1,200            |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|         Scikit-Learn ColumnTransformer Pipeline              |
|  - Numerical (7): SimpleImputer(median) + StandardScaler     |
|  - Categorical (4): SimpleImputer(mode) + OneHotEncoder      |
|           * FITTED EXCLUSIVELY ON TRAINING SET *             |
+--------------------------------------------------------------+
                               |
         +---------------------+---------------------+
         |                     |                     |
         v                     v                     v
+--------------------+ +---------------+ +--------------------+
| Logistic Regr.     | | Decision Tree | |  Random Forest     |
| (Class-Weighted)   | | (Depth = 5)   | | (100 Estimators)   |
+--------------------+ +---------------+ +--------------------+
         |                     |                     |
         +---------------------+---------------------+
                               |
                               v
+--------------------------------------------------------------+
|        Held-Out Test Set Evaluation (N = 1,200)              |
|   Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|    Joblib Persistence (all_models.joblib & metadata.json)    |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|   STREAMSCOPE Cinematic Streamlit Web Application (Port 8501)|
|    - Section A: Overview Dashboard                           |
|    - Section B: Subscriber Prediction Tool                   |
|    - Section C: Viewer Engagement Analytics                  |
|    - Section D: Model Lab & Benchmarking                     |
|    - Section E: Retention Insights & Playbook                |
+--------------------------------------------------------------+
```

---

### 6. MATHEMATICAL FORMULATIONS OF MACHINE LEARNING ALGORITHMS

#### 6.1 Logistic Regression
Logistic Regression models the posterior probability of subscriber churn $Y \in \{0, 1\}$ using the logistic sigmoid function:
$$P(Y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

The model is optimized by minimizing the regularized negative log-likelihood (binary cross-entropy):
$$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

#### 6.2 Decision Tree Classifier
The Decision Tree partitions the subscriber feature space recursively by identifying splits $s$ on feature $j$ that maximize Gini Impurity reduction:
$$Gini(D) = 1 - \sum_{k=0}^{1} p_k^2$$
$$\Delta Gini(D, j, s) = Gini(D) - \frac{|D_L|}{|D|} Gini(D_L) - \frac{|D_R|}{|D|} Gini(D_R)$$

To avoid memorizing statistical noise, structural regularization constraints were applied: `max_depth=5`, `min_samples_split=10`, and `min_samples_leaf=5`.

#### 6.3 Random Forest Classifier
Random Forest is an ensemble bagging technique that trains $B = 100$ decorrelated decision trees on bootstrap resamples $\mathcal{D}_b$. At each split node, a random subset of $m = \sqrt{p}$ features is sampled.

The ensemble predicted probability is the arithmetic mean across all trees:
$$\hat{P}(Y=1|\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} P_b(Y=1|\mathbf{x})$$

This ensemble averaging drastically lowers model variance without inflating bias.

---

### 7. DATA PREPROCESSING & DATA LEAKAGE PREVENTION

1. **Identifier Removal:** `customerID` was removed prior to modeling to prevent artificial memorization.
2. **Missing-Value Imputation:** 
   - Numerical columns: Imputed with the **median** of the training split.
   - Categorical columns: Imputed with the **most frequent** category of the training split.
3. **Feature Scaling & Encoding:**
   - Numerical features: Standardized using `StandardScaler` ($z = (x - \mu) / \sigma$).
   - Categorical features: Encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')`.
4. **Data Leakage Guarantee:** The entire transformation process is contained within a Scikit-Learn `ColumnTransformer` fitted **exclusively on the training dataset (`X_train`)**. The test set and user inputs during live inference are transformed strictly using training parameters.

---

### 8. EXPERIMENTAL SETUP AND TRAINING PIPELINE
- **Programming Language:** Python 3.14
- **Libraries:** Scikit-Learn 1.8.0, Pandas 3.0.2, NumPy 2.4.4, Plotly 7.1.0, Streamlit 1.66.1, Joblib 1.5.3
- **Reproducibility:** Seed fixed at `random_state=42`.
- **Handling Class Imbalance:** In streaming subscription services, churn is typically the minority class (~26.7%). Rather than artificially oversampling the dataset, cost-sensitive class weights were utilized:
  $$w_j = \frac{N}{2 \cdot N_j}$$
  This automatically penalizes false negatives on the churn class during loss optimization.

---

### 9. EXPERIMENTAL RESULTS & ALGORITHMIC BENCHMARKING

All three models were evaluated on the **identical held-out test dataset** ($N = 1,200$, consisting of 320 actual Churners and 880 actual Retained subscribers):

| Algorithm | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **73.25%** | **0.4990** | **74.69%** | **0.5982** | **0.7991** |
| **Random Forest** | **74.50%** | **0.5168** | **67.19%** | **0.5842** | **0.7899** |
| **Decision Tree** | 71.25% | 0.4682 | 57.50% | 0.5161 | 0.7508 |

#### Confusion Matrix Breakdown (Held-Out Test Set $N = 1,200$):
- **Logistic Regression:**
  - True Negatives (TN): 640 | False Positives (FP): 240
  - False Negatives (FN): 81 | True Positives (TP): 239 (Recall: 74.69%)
- **Random Forest:**
  - True Negatives (TN): 679 | False Positives (FP): 201
  - False Negatives (FN): 105 | True Positives (TP): 215 (Recall: 67.19%)
- **Decision Tree:**
  - True Negatives (TN): 671 | False Positives (FP): 209
  - False Negatives (FN): 136 | True Positives (TP): 184 (Recall: 57.50%)

#### Technical Evaluation Rationale:
- **Misleading Nature of Accuracy:** A naive baseline model predicting "Retained" for every subscriber achieves **73.3% Accuracy**, yet identifies **zero churning subscribers (0% Recall)**.
- **Cost Asymmetry:** In streaming economics, missing an at-risk subscriber (**False Negative**) results in the loss of recurring subscription revenue ($120–$240 annually). Conversely, sending a re-engagement email or trailer recommendation to a loyal customer (**False Positive**) has negligible marginal cost.
- **Selection Decision:** **Logistic Regression** and **Random Forest** both provide strong predictive power; Logistic Regression was selected as champion for achieving the highest **Recall (74.69%)** and **ROC-AUC (0.7991)**.

---

### 10. FEATURE IMPORTANCE & BEHAVIORAL DRIVERS

Feature importances extracted from the trained Random Forest revealed the top attributes driving subscriber attrition:

| Feature Name | Clean Label | Relative Weight | Business Significance |
| :--- | :--- | :---: | :--- |
| `DaysSinceLastActive` | Days Since Last Active Session | **26.4%** | Inactivity beyond 14 days causes a steep rise in cancellation hazard. |
| `AverageWeeklyHours` | Weekly Viewing Hours | **21.8%** | Subscribers watching under 8 hrs/week experience double the churn of 15+ hr viewers. |
| `TenureMonths` | Subscriber Lifecycle (Months) | **17.2%** | Months 1 to 6 exhibit the highest drop-off rate before habit formation stabilizes. |
| `NumberOfProfiles` | Profiles on Account | **11.5%** | Single profile accounts churn at 33.5% vs <18% for 4-profile household accounts. |
| `MonthlyCharges` | Monthly Plan Fee ($) | **8.1%** | Price sensitivity influences cancellation when combined with low watch time. |
| `CustomerSupportTickets`| Support Complaints | **6.4%** | 2+ tickets for playback or billing issues accelerates cancellation. |
| `ContentDownloadsPerMonth`| Offline Downloads | **4.2%** | Mobile/tablet downloaders show higher platform stickiness and lower churn. |

---

### 11. STREAMSCOPE USER INTERFACE DESIGN SYSTEM

The user interface was built using Streamlit and Plotly following a **cinematic editorial design system**:
- **Palette:** Deep charcoal (`#0d0e12`), graphite main surfaces (`#181a20`), muted crimson accents (`#d9232e`), soft white typography (`#f3f4f6`).
- **Brand Wordmark:** `◬ STREAMSCOPE` | *Subscriber Intelligence Platform*.
- **Horizontal Metric Strip:** Compact telemetry displays for subscribers monitored (6,000), retention rate (73.3%), churn rate (26.7%), revenue at risk ($24.8k/mo), and mean watch time (15.5 hrs/wk).
- **Five Application Sections:**
  1. *Section A: Overview Dashboard* (Donut chart, plan tiers, inactivity hazard curve, device usage).
  2. *Section B: Subscriber Churn Prediction* (Interactive form, presets for high-risk and loyal personas, semi-circular risk dial gauge, and explainability breakdown).
  3. *Section C: Viewer Engagement Analytics* (Overlaid weekly watch density, offline download impact, household profile lock-in).
  4. *Section D: Model Lab & Benchmarking* (Multi-metric comparison bar chart, annotated confusion matrix heatmap, overlaid ROC curves).
  5. *Section E: Retention Insights & Playbook* (Horizontal feature importances, structured retention intervention matrix, and correlation vs. causation notice).

---

### 12. STRATEGIC RETENTION PLAYBOOK & CAUSALITY BOUNDARIES

#### Strategic Retention Matrix:
| Risk Profile | Observed Pattern | Intervention Strategy | Target Timing | Business Goal |
| :--- | :--- | :--- | :--- | :--- |
| **Dormant Viewer** | 14+ days inactive | Personalized "New Season / Top 10 for You" notification | Day 10–14 of inactivity | Re-establish habit loop before billing cycle |
| **Low-Engagement Streamer** | <8 hrs/week watch time | Curated genre carousels and "Quick 20-min Watch" collections | Weekly digest | Increase daily playback frequency |
| **New Subscriber** | Months 1 to 3 | Interactive onboarding tour, prompt profile creation & download guide | First 14 days | Drive multi-device adoption |
| **Single-Profile Basic User**| 1 profile, Basic tier | Free 30-day promotional upgrade to Standard (2 screens) | Month 2 renewal | Establish household lock-in |
| **Technical Friction Account** | 2+ support tickets | Priority concierge check-in + 1-month $5 courtesy billing credit | Within 24h of 2nd ticket | Neutralize platform dissatisfaction |

#### Correlation vs. Causation Methodological Distinction:
Machine learning models identify **statistical correlations and predictive associations**, not deterministic causal mechanisms. For example, while high *Days Since Last Active* correlates with churn, inactivity is a *symptom* of declining viewer interest, not the sole mechanical cause. Product interventions must be validated via randomized controlled trials (A/B testing) before establishing definitive cause-and-effect.

---

### 13. CONCLUSION AND FUTURE WORK

#### 13.1 Conclusion
The **STREAMSCOPE** project successfully demonstrates an end-to-end Machine Learning intelligence platform for streaming subscriber retention:
1. Formulated a realistic streaming subscriber telemetry dataset capturing viewing hours, inactivity, and plan tiers.
2. Built leak-free Scikit-Learn pipelines evaluating three classification models on held-out test data.
3. Achieved **74.69% Recall** on the churn class and an **ROC-AUC of 0.7991**, effectively prioritizing the detection of at-risk subscribers.
4. Delivered a production-grade, cinematic web interface featuring interactive Plotly visualizations and persona simulations.

#### 13.2 Future Work
1. **Survival Analysis:** Implement Kaplan-Meier and Cox Proportional Hazards models to estimate the *expected remaining customer lifetime in months*.
2. **Local Explainable AI (SHAP):** Integrate SHAP force plots to explain individual predictions on a per-subscriber basis.
3. **Automated Marketing Webhooks:** Connect model outputs to CRM platforms (e.g. Braze, HubSpot) to automatically dispatch re-engagement push notifications when a subscriber crosses a 60% churn probability threshold.

---

### 14. REFERENCES
1. Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix recommender system: Algorithms, business value, and innovation. *ACM Transactions on Management Information Systems (TMIS)*, 6(4), 1-19.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
4. Burez, J., & Van den Poel, D. (2009). Handling class imbalance in customer churn prediction. *Expert Systems with Applications*, 36(3), 4626-4636.
5. Streamlit Framework Documentation (2026).
