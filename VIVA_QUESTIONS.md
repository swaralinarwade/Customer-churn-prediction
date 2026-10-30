# COMPREHENSIVE VIVA VOCE QUESTIONS & ANSWERS
## STREAMSCOPE: Subscriber Intelligence Platform
### Netflix-Inspired Streaming Customer Churn Prediction & Viewer Engagement Analytics
**Third-Year Computer Engineering Mini-Project | Machine Learning Subject**

This document provides technically rigorous, concise, and examiner-approved answers to all common questions asked during academic viva examinations, project defenses, and practical evaluations.

---

### SECTION 1: MACHINE LEARNING FUNDAMENTALS & STREAMING DOMAIN

#### Q1: What is the main problem your mini-project solves?
**Answer:** In subscription video-on-demand (SVOD) streaming platforms (like Netflix), acquiring new subscribers costs 5 to 7 times more than retaining existing ones. Because subscribers can cancel anytime with zero friction, platforms suffer recurring revenue loss when viewing engagement drops. **STREAMSCOPE** uses machine learning to identify at-risk subscribers weeks in advance by tracking telemetry such as days of inactivity, falling watch hours, and plan tiers, allowing operators to launch targeted, cost-effective retention campaigns.

#### Q2: What type of machine learning task is subscriber churn prediction?
**Answer:** It is a **Supervised Binary Classification** task:
- **Supervised:** The model learns from historical subscriber accounts where the ground truth outcome (`Churn = 0` or `1`) is known.
- **Binary Classification:** The target variable belongs to exactly one of two mutually exclusive classes: `0` (Retained) or `1` (Churned).

#### Q3: What dataset did you use, and what is its class distribution?
**Answer:** We used an educational streaming subscriber dataset of **6,000 subscriber records** across **13 columns**. The class distribution is:
- **Retained (`0`):** 4,399 subscribers (~73.32%)
- **Churned (`1`):** 1,601 subscribers (~26.68%)
This represents an **imbalanced classification problem** (~2.75 : 1 ratio).

---

### SECTION 2: DATA PREPROCESSING & DATA LEAKAGE PREVENTION

#### Q4: Why did you drop the `customerID` column?
**Answer:** `customerID` is an arbitrary unique string identifier (e.g., `STR-10042`). It has zero generalized predictive capability. If left in the feature matrix, a non-linear algorithm like a Decision Tree could memorize individual ID tokens, leading to severe overfitting.

#### Q5: How did you handle missing values in the dataset?
**Answer:** We implemented a Scikit-Learn `ColumnTransformer` with dedicated `SimpleImputer` transformers:
- **Numerical features** (`AverageWeeklyHours`, `ContentDownloadsPerMonth`): Imputed using the **median** of the training split to be robust against outliers.
- **Categorical features**: Imputed using the **most frequent** category of the training split.

#### Q6: What is Data Leakage, and how did you prevent it?
**Answer:** **Data leakage** occurs when information from the test dataset is inadvertently shared with the training process, producing artificially inflated, ungeneralizable evaluation metrics.
We prevented data leakage by:
1. Executing a **stratified 80/20 train/test split** *before* any feature transformation.
2. Fitting the `SimpleImputer`, `StandardScaler` (means and standard deviations), and `OneHotEncoder` (categories) **strictly on `X_train`**.
3. Applying `.transform()` to `X_test` and live user inference using the parameters frozen during the training phase.

#### Q7: Why did you use `StandardScaler` on numerical features?
**Answer:** Features such as `TenureMonths` (1 to 72) and `DaysSinceLastActive` (0 to 45) operate on completely different scales than `MonthlyCharges` or `CustomerSupportTickets`. Distance-based and gradient-descent algorithms (such as Logistic Regression) are sensitive to feature scales; without standardization, features with larger ranges dominate the loss gradients. `StandardScaler` standardizes each feature to have a mean of 0 and a variance of 1:
$$z = \frac{x - \mu}{\sigma}$$

#### Q8: Why use One-Hot Encoding instead of Label Encoding for categorical variables?
**Answer:** Label encoding assigns arbitrary integers (0, 1, 2) to categories. For nominal variables like `PrimaryDevice` (Smart TV, Mobile, Desktop, Tablet), label encoding introduces an artificial numerical order ($3 > 2 > 1 > 0$) that misleads linear models. One-Hot Encoding creates independent binary dummy columns, preserving equal geometric distance between distinct categories.

---

### SECTION 3: MODEL EVALUATION & IMBALANCED DATA

#### Q9: Why is raw Accuracy misleading for subscription churn?
**Answer:** In a dataset with 73.3% retained subscribers and 26.7% churners:
- A naive baseline classifier that blindly predicts *"Retained"* for every single subscriber achieves an impressive-sounding **73.32% Accuracy**.
- However, that model has **0% Recall** on churners and is completely useless to streaming operators because it fails to catch a single customer at risk of canceling.
Therefore, Accuracy alone fails to reflect model performance on imbalanced data.

#### Q10: Why do we prioritize Recall for the Churn class over Precision in SVOD?
**Answer:** Due to **Cost Asymmetry**:
- **False Negative (FN):** Failing to identify an at-risk subscriber results in the loss of that customer and their recurring annual subscription revenue ($120–$240).
- **False Positive (FP):** Flagging a loyal subscriber for a retention recommendation (e.g., an automated trailer preview or genre recommendation email) carries essentially zero marginal cost.
Because False Negatives are substantially more expensive than False Positives, we optimize for **Recall on Churn** ($\frac{TP}{TP + FN}$).

#### Q11: What is the F1-Score, and why is it useful?
**Answer:** The F1-Score is the **harmonic mean** of Precision and Recall:
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
The harmonic mean severely penalizes extreme trade-offs, ensuring that a high recall score is not artificially obtained by predicting churn for 100% of subscribers.

#### Q12: What does the ROC-AUC score represent?
**Answer:** 
- The **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate (Recall) against the False Positive Rate ($1 - \text{Specificity}$) across all classification thresholds from 0.0 to 1.0.
- **ROC-AUC (Area Under the Curve)** measures the probability that the classifier will assign a higher churn probability to a randomly chosen churned subscriber than to a randomly chosen retained subscriber.
- An AUC of **0.50** equals random coin flipping; an AUC of **1.00** equals perfect discrimination. Our champion model achieved an AUC of **0.7991**, indicating strong discriminative capability.

#### Q13: Explain your model's Confusion Matrix on the test set.
**Answer:** On the held-out test set of 1,200 subscribers (320 Churners, 880 Retained):
- **True Negatives (TN = 640):** Correctly identified retained subscribers.
- **False Positives (FP = 240):** Retained subscribers flagged for retention inspection.
- **False Negatives (FN = 81):** Churners missed by the model.
- **True Positives (TP = 239):** Churners successfully intercepted!
- **Recall on Churn:** $\frac{239}{239 + 81} = \mathbf{74.69\%}$.

---

### SECTION 4: ALGORITHMIC COMPARISON & SELECTION

#### Q14: How does Logistic Regression perform binary classification?
**Answer:** Logistic Regression applies the **sigmoid activation function** to a linear combination of input features to output a calibrated probability between 0 and 1:
$$P(Y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
It is trained by minimizing the binary cross-entropy log-loss function using Quasi-Newton optimization (`lbfgs`).

#### Q15: What is the difference between a Decision Tree and a Random Forest?
**Answer:**
- A **Decision Tree** is a single hierarchical model that splits data recursively based on Gini impurity. It is prone to high variance and easily overfits training data.
- A **Random Forest** is an **ensemble bagging** algorithm that builds 100 decorrelated decision trees in parallel. Each tree is trained on a random bootstrap sample of the data, and each split evaluates only a random subset of features ($\sqrt{p}$). The final probability is the ensemble average across all trees, which drastically reduces variance and prevents overfitting.

#### Q16: How did you handle class imbalance in model training?
**Answer:** We set `class_weight='balanced'` in all three Scikit-Learn estimators:
$$w_j = \frac{N}{2 \cdot N_j}$$
This automatically assigns higher penalties (~2.75x) to misclassifications on the minority churn class during loss optimization, without modifying the underlying training distribution prior to train/test partitioning.

#### Q17: Which model was selected as your final default, and why?
**Answer:** **Logistic Regression** was selected as the champion model because it achieved the highest **Recall on Churn (74.69%)**, the highest **F1-Score (0.5982)**, and the highest **ROC-AUC (0.7991)** on the held-out test dataset, delivering the strongest balance between catching at-risk subscribers and maintaining high overall accuracy (73.25%).

---

### SECTION 5: EXPLAINABILITY & STREAMING DOMAIN INSIGHTS

#### Q18: What are the top features driving subscriber churn in your model?
**Answer:**
1. **Days Since Last Active (26.4%):** Strongest indicator; inactivity beyond 14 days causes a steep rise in cancellation hazard.
2. **Weekly Viewing Hours (21.8%):** Viewers watching under 8 hrs/week have double the churn rate of active streamers.
3. **Subscriber Tenure (17.2%):** First 6 months exhibit the steepest drop-off hazard before habit loops solidify.
4. **Profiles on Account (11.5%):** Accounts with multiple profiles experience lower churn due to shared household lock-in.

#### Q19: Does your model prove that a feature *causes* churn?
**Answer:** **No.** Machine learning models identify **statistical correlations and predictive associations**, not causal proof. 
For example, while long *Days Since Last Active* is strongly correlated with churn, inactivity is a *symptom* of declining viewer interest, not the sole mechanical cause. Establishing causality requires controlled A/B testing before launching major retention interventions.

#### Q20: How does STREAMSCOPE handle live predictions in the web UI?
**Answer:**
1. When a user enters subscriber attributes in the web form, the data is structured into a single-row Pandas DataFrame matching the 11 feature schema.
2. The DataFrame is passed directly to the persisted Scikit-Learn `Pipeline` loaded via `joblib`.
3. The pipeline's `ColumnTransformer` scales numerical features and one-hot encodes categorical features using pre-computed training parameters.
4. The classifier outputs `predict()` (class 0 or 1) and `predict_proba()` (probability percentage).
5. The UI renders the result banner, semi-circular Plotly risk gauge, and explainable risk driver breakdown in real time.
