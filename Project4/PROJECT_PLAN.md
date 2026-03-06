# Project 4 Plan: Choose Your Own Dataset

## Dataset Options

### Option 1: Online News Popularity (UCI ID=332) -- RECOMMENDED
**Source:** University of California Irvine ML Repository

**Description:** 39,644 Mashable articles with 58 features including word counts, sentiment scores,
topic tags, and day of publication. The target is the number of shares.

**Goal (inferential):** Determine what article characteristics drive viral sharing. With 58 features,
this dataset is well-suited for identifying the most important factors behind article popularity.

**Why it works:**
- 39,644 rows and 58 features, well above the minimums
- Regression target (shares) with a clear real-world interpretation
- Rich feature set spanning NLP metrics, timing, and topic categories
- Decision tree or random forest provides interpretable feature importance rankings

---

### Option 2: Default of Credit Card Clients (UCI ID=350)
**Source:** University of California Irvine ML Repository

**Description:** 30,000 Taiwanese credit card holders with 23 features covering payment history,
bill amounts, and demographics. The target is whether the holder defaults the following month.

**Goal (predictive):** Build a classifier to flag high-risk customers before they default.

**Why it works:**
- 30,000 rows and 23 features, meets all requirements
- Binary classification with real financial stakes
- Clean data with no missing values
- Random forest provides strong predictive accuracy and feature importance

---

### Option 3: Electrical Grid Stability (UCI ID=471)
**Source:** University of California Irvine ML Repository

**Description:** 10,000 simulated power grid configurations with 12 input features (power consumption
rates, reaction times) predicting grid stability as either a binary label or a continuous score.

**Goal (predictive):** Predict whether a given power grid configuration will remain stable.

**Why it works:**
- 10,000 rows and 12 features, meets minimum requirements
- Very clean data with no missing values and no categorical encoding needed
- Fast training times allow for easy experimentation
- Can be framed as classification or regression

---

## Execution Plan

The following steps apply to whichever dataset is selected.

### Step 1: Q1a -- Motivation and Goal
- Write a motivation paragraph grounded in the dataset context
- State the goal clearly as either predictive or inferential
- Identify the target variable and why it is meaningful

### Step 2: Q1b -- Variable Descriptions
- Group related variables (e.g., time variables, content metrics, weather conditions)
- Briefly explain the role of each group in the model
- Note the target variable separately

### Step 3: Task 1 -- Load, Preprocess, Split
- Fetch the dataset via `ucimlrepo` (no manual download needed)
- Define and extract the target column
- Drop non-feature columns (IDs, leakage columns, etc.)
- Encode categorical variables if needed
- Perform an 80/20 train/test split using `train_test_split`
- Return `X_train, X_test, y_train, y_test`

### Step 4: Task 2 -- Unsupervised Learning
- Standardize features before PCA
- Fit `PCA(n_components=2)` on training data, transform both train and test
- Fit `KMeans(k=3)` on the 2D PCA-transformed training data
- Produce two scatter plots:
  - Points colored by cluster label
  - Points colored by target value
- Print the top loading variable for each principal component

### Step 5: Unsupervised Questions
- Answer whether KMeans clusters align with the target (a)
- Name the variables with the highest loadings in PC1 and PC2 (b)

### Step 6: Task 3 -- Supervised Model
- Select model based on goal type:
  - Inferential goal: Decision Tree Regressor/Classifier (interpretable structure)
  - Predictive goal: Random Forest Regressor/Classifier (higher accuracy)
- Implement `build_model`, `evaluate_predictions`, and `train_eval`
- Include a visualization: decision tree plot, feature importance bar chart,
  or predicted vs. actual scatter plot
- Report RMSE and R2 for regression, or accuracy and a confusion matrix for classification

### Step 7: Results
- Relate model performance back to the original goal
- Describe what the model reveals (key features, decision boundaries, etc.)
- Comment on model quality and any limitations
- Suggest practical takeaways based on findings
