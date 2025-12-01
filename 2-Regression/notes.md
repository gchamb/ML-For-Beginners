## Chapter 1: Regression Types

`Linear Regression` - A type of regression technique that is used when your target (aka output) is a numeric value.

> Ex. What is the height of this person

`Logistic Regression` - A type of regression technique that is used when your target (aka output) is a category assignment.

> Ex. Is this food more for vegans or meat eaters?

`Hold out validation` - The process of splitting a dataset into a certain ratio of training and testing dataset using Scikit Learn's `train_test_split()` method/function is called:

## Chapter 3: Linear Regression

### Key Terminology

`Basic Linear Regression` - A regression technique that models the relationship between a dependent variable (Y) and one or more independent variables (X) using a straight line. The equation is Y = a + bX, where 'a' is the y-intercept and 'b' is the slope.

`Polynomial Regression` - A form of regression that models the relationship between variables using polynomial equations (curves instead of straight lines). It can capture non-linear relationships by including squared, cubed, or higher-degree terms of the input variables.

`Features (X)` - The input variables or independent variables used to make predictions. Also called explanatory variables or predictors.

`Target (Y)` - The output variable or dependent variable that we want to predict. Also called the label or response variable.

`Training Data` - The subset of data used to train the model and learn the relationship between features and target.

`Test Data` - The subset of data used to evaluate the model's performance on unseen data, typically 20-30% of the total dataset.

`Least-Squares Regression` - A mathematical method that finds the best-fitting line by minimizing the sum of squared differences between actual and predicted values.

`Line of Best Fit` - The regression line that best represents the relationship in the data, positioned to minimize the distance from all data points.

`Slope (b)` - The coefficient that indicates how much Y changes for each unit increase in X. Calculated as the change in Y divided by the change in X.

`Y-Intercept (a)` - The value of Y when X equals zero. The point where the regression line crosses the Y-axis.

`Correlation Coefficient (r)` - A statistical measure (ranging from -1 to +1) that indicates the strength and direction of the linear relationship between two variables.

- **r = +1**: Perfect positive correlation
- **r = -1**: Perfect negative correlation
- **r = 0**: No linear correlation
- **|r| > 0.7**: Strong correlation
- **0.3 < |r| < 0.7**: Moderate correlation
- **|r| < 0.3**: Weak correlation

`Categorical Features` - Variables that represent categories or groups (e.g., variety, city, package type) rather than numeric values.

`Residual (Error)` - The difference between actual and predicted values: Error = Actual - Predicted

`Model Determination` - See R² Score (Coefficient of Determination)

## Statistics & Metrics Used in Linear Regression

### Error Metrics

`Mean Squared Error (MSE)` - The average of squared differences between actual and predicted values. Measures the average magnitude of errors.

**Formula:** `MSE = (1/n) × Σ(yi - ŷi)²`

Where:

- n = number of observations
- yi = actual values
- ŷi = predicted values
- Σ = sum of all values

**Interpretation:**

- Lower MSE = Better model performance
- MSE = 0 means perfect predictions
- MSE is in squared units of the target variable

---

`Root Mean Squared Error (RMSE)` - The square root of Mean Squared Error. Provides error in the same units as the target variable.

**Formula:** `RMSE = √MSE = √[(1/n) × Σ(yi - ŷi)²]`

**Interpretation:**

- Same units as the target variable (e.g., dollars for price)
- Easier to interpret than MSE
- Lower RMSE = Better model
- Can be expressed as percentage: `(RMSE / mean) × 100%`

---

`R² Score (Coefficient of Determination)` - Indicates the proportion of variance in the dependent variable that is predictable from the independent variables.

**Formula:** `R² = 1 - (SS_res / SS_tot)`

Where:

- SS_res = Σ(yi - ŷi)² (sum of squared residuals)
- SS_tot = Σ(yi - ȳ)² (total sum of squares)
- ȳ = mean of actual values

**Interpretation:**

- **R² = 1.0**: Perfect predictions (100% of variance explained)
- **R² = 0.0**: Model doesn't explain any variance (predictions = mean)
- **R² < 0.0**: Model performs worse than using the mean
- **R² = 0.97**: Model explains 97% of the variance (excellent)
- **R² = 0.70-0.90**: Good model
- **R² = 0.50-0.70**: Moderate model
- **R² < 0.50**: Weak model

---

### Correlation Metrics

`Pearson Correlation Coefficient (r)` - Measures the strength and direction of linear relationship between two variables.

**Formula:** `r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]`

**Alternative Formula:** `r = (n×Σ(xi×yi) - Σxi×Σyi) / √[(n×Σxi² - (Σxi)²) × (n×Σyi² - (Σyi)²)]`

Where:

- xi, yi = individual x and y values
- x̄, ȳ = mean of x and y values
- n = number of data points
- Σ = sum of

**Interpretation:**

- Values range from -1 to +1
- Positive values indicate positive correlation
- Negative values indicate negative correlation
- Used to determine if training a predictive model makes sense

## Definitions

`Supervised Learning` - Learning from labeled data to map to known outputs. Learns the relationship between inputs and outputs in order to predict new, unseen data.

`Unsupervised Learning` - Learning from unlabeled where the algorithm discovers patterns on its own.

## Chapter 4: Logistic Regression

### Key Terminology

`Logistic Regression` - A linear-based classification method used to predict binary categories (e.g., white or not white, spam or not spam). Despite the name "regression," it's actually a classification technique.

`Binary Classification` - Prediction between two categories (0 or 1, True or False, Yes or No). Example: predicting if a pumpkin is white or orange.

`Multinomial Classification` - Classification involving more than two categories. Example: "Orange, White, and Striped" pumpkins.

`Ordinal Classification` - Classification with ordered categories that follow a logical sequence. Example: pumpkin sizes (mini, small, medium, large, xl, xxl).

`Sigmoid Function (Logistic Function)` - An S-shaped curve that maps any input value to a value between 0 and 1. Used to convert linear regression output into probabilities for classification.

**Formula:** `σ(x) = 1 / (1 + e^(-x))`

**Properties:**

- Output range: 0 to 1
- S-shaped curve
- Midpoint at x = 0 (where output = 0.5)
- Used to determine class: if σ(x) > 0.5 → class 1, otherwise → class 0

`Maximum Likelihood` - The principle used by logistic regression to find the best-fitting model by maximizing the probability of observing the actual data.

### Feature Encoding

`Feature Encoding` - The process of converting categorical (text) data into numerical format that machine learning algorithms can process.

`Ordinal Encoder` - Encodes ordinal variables (categories with logical ordering) by assigning sequential numbers based on the order.

**Example:**

```python
Item Size: ['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']
Encoded as: [0, 1, 2, 3, 4, 5, 6]
```

`One-Hot Encoder (Categorical Encoder)` - Encodes nominal variables (categories without logical ordering) by creating binary columns for each category.

**Example:**

```
Variety: ['Pie Type', 'Carving', 'Mini']

One-Hot Encoded:
Variety_Pie Type: [1, 0, 0]
Variety_Carving:  [0, 1, 0]
Variety_Mini:     [0, 0, 1]
```

`Label Encoder` - Encodes target labels into values between 0 and n_classes-1.

**Example:**

```
Color: ['ORANGE', 'WHITE', 'ORANGE', 'WHITE']
Encoded: [0, 1, 0, 1]
```

`ColumnTransformer` - A scikit-learn utility that combines multiple encoders and applies them to appropriate columns in a single step.

### Model Evaluation Metrics

`Confusion Matrix` - A table showing the model's true vs. false positives and negatives, used to evaluate classification accuracy.

**Structure:**

```
                Predicted
                0       1
Actual    0    TN      FP
          1    FN      TP
```

Where:

- **TN (True Negative)**: Correctly predicted as class 0
- **FP (False Positive)**: Incorrectly predicted as class 1 (Type I error)
- **FN (False Negative)**: Incorrectly predicted as class 0 (Type II error)
- **TP (True Positive)**: Correctly predicted as class 1

**Example:**

```
Confusion Matrix:
[[162,   4],    # 162 true negatives, 4 false positives
 [ 11,  22]]    # 11 false negatives, 22 true positives
```

`Precision` - The fraction of correct positive predictions out of all positive predictions.

**Formula:** `Precision = TP / (TP + FP)`

**Example:** `Precision = 22 / (22 + 4) = 0.846`

**Interpretation:**

- Answers: "Of all items predicted as positive, how many are actually positive?"
- High precision = Few false positives
- Important when false positives are costly

`Recall (Sensitivity, True Positive Rate)` - The fraction of actual positives that were correctly identified.

**Formula:** `Recall = TP / (TP + FN)`

**Example:** `Recall = 22 / (22 + 11) = 0.667`

**Interpretation:**

- Answers: "Of all actual positive items, how many did we correctly identify?"
- High recall = Few false negatives
- Important when false negatives are costly (e.g., disease detection)

`F1-Score` - The harmonic mean of precision and recall, providing a single score that balances both metrics.

**Formula:** `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Example:** `F1 = 2 × (0.846 × 0.667) / (0.846 + 0.667) = 0.746`

**Interpretation:**

- Range: 0 to 1
- F1 = 1: Perfect precision and recall
- F1 = 0: Worst possible score
- Useful when you need balance between precision and recall

`Accuracy` - The percentage of all predictions that were correct.

**Formula:** `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

**Example:** `Accuracy = (22 + 162) / (22 + 162 + 4 + 11) = 0.925 (92.5%)`

**Interpretation:**

- Simple overall performance metric
- Can be misleading with imbalanced datasets

`Support` - The number of actual occurrences of each class in the dataset.

**Example:**

```
Class 0 (Orange): 166 samples
Class 1 (White): 33 samples
```

`Macro Average` - The unweighted mean of metrics for each class, treating all classes equally regardless of support.

**Calculation:** `Macro Avg = (Metric_Class0 + Metric_Class1) / 2`

`Weighted Average` - The mean of metrics for each class, weighted by the number of samples (support) in each class.

**Calculation:** `Weighted Avg = (Metric_Class0 × Support0 + Metric_Class1 × Support1) / Total_Samples`

### ROC Curve Analysis

`ROC Curve (Receiver Operating Characteristic)` - A graph showing the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR) at different classification thresholds.

**Axes:**

- X-axis: False Positive Rate (FPR) = FP / (FP + TN)
- Y-axis: True Positive Rate (TPR) = TP / (TP + FN) = Recall

**Interpretation:**

- Diagonal line (y=x): Random classifier (baseline)
- Curve above diagonal: Better than random
- Steeper curve = Better model
- More area under curve = Better performance

`AUC (Area Under the Curve)` - A single number (0 to 1) representing the total area under the ROC curve.

**Interpretation:**

- **AUC = 1.0**: Perfect classifier
- **AUC = 0.9-1.0**: Excellent model
- **AUC = 0.8-0.9**: Good model
- **AUC = 0.7-0.8**: Fair model
- **AUC = 0.5**: No discrimination (random classifier)
- **AUC < 0.5**: Worse than random (model is inverted)

**Example:** `AUC = 0.975` indicates an excellent model with 97.5% probability of correctly distinguishing between classes.

### Key Differences: Linear vs. Logistic Regression

| Aspect           | Linear Regression                       | Logistic Regression                             |
| ---------------- | --------------------------------------- | ----------------------------------------------- |
| **Output Type**  | Continuous values                       | Binary categories (0 or 1)                      |
| **Use Case**     | Predict quantities (price, temperature) | Predict classifications (yes/no, spam/not spam) |
| **Output Range** | -∞ to +∞                                | 0 to 1 (probabilities)                          |
| **Function**     | Linear equation: y = mx + b             | Sigmoid function: σ(x) = 1/(1+e^(-x))           |
| **Correlation**  | Works better with correlated features   | Features don't need to correlate                |
| **Evaluation**   | R², MSE, RMSE                           | Precision, Recall, F1-score, AUC                |

### Data Requirements for Logistic Regression

1. **More data is better** - Logistic regression gives more accurate results with larger datasets
2. **Clean data required** - Remove null values and handle missing data properly
3. **Features don't need to correlate** - Unlike linear regression, weak correlations are acceptable
4. **Binary or categorical target** - Target variable must be categories, not continuous values

### Example Use Cases

**Binary Classification:**

- Email: Spam or Not Spam
- Medical: Disease Present or Not Present
- Finance: Loan Default or Not Default
- Pumpkin: White or Orange

**Multinomial Classification:**

- Pumpkin variety: Pie Type, Carving, or Mini
- Animal type: Dog, Cat, or Bird
- Product category: Electronics, Clothing, or Food

**Ordinal Classification:**

- Customer satisfaction: Poor, Fair, Good, Excellent
- Education level: High School, Bachelor's, Master's, PhD
- Pumpkin size: Small, Medium, Large, Extra Large
