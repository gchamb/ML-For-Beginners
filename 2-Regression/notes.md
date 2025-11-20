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
