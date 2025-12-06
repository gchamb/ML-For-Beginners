# Classification

`Classification` - is a form of supervised learning that typically fall into two groups:

- Binary Classification
- Multiclass Classification

### Balancing the data

When the classes in your dataset are unevenly represented (class imbalance), models can become biased
towards the majority class and perform poorly on minority classes. For example, if 90% of recipes are
`korean` and only 3% are `thai`, a naive classifier could achieve high overall accuracy by always
predicting `korean` while completely failing to recognize `thai` recipes. Balancing the dataset helps
ensure the model learns patterns for all classes and results in fairer, more useful predictions.

Why balance data?

- **Avoid biased predictions:** Prevents the model from favouring majority classes.
- **Improve recall on minority classes:** Helps detect less frequent but important classes.
- **More reliable evaluation:** Makes metrics like precision, recall, and F1-score more meaningful.

Common strategies for balancing

- **Undersampling:** Remove examples from the majority class. Simple but can discard useful data.
- **Oversampling:** Duplicate examples from minority classes. May cause overfitting if naively applied.
- **SMOTE (Synthetic Minority Over-sampling Technique):** Create new synthetic examples for
  minority classes by interpolating between existing samples. Balances data without exact duplicates
  and usually reduces overfitting compared to naive oversampling.

---

## Terminology & Concepts (Lesson 2: Classifiers)

### Core ML Terms

**Model / Estimator**

- The algorithm that learns patterns from training data
- Examples: `LogisticRegression`, `SVC`, `DecisionTreeClassifier`
- After calling `.fit()`, the model contains learned parameters

**Parameters vs Hyperparameters**

- **Parameters**: Values the model _learns_ during training (e.g., logistic regression coefficients/weights)
- **Hyperparameters**: Settings you choose _before_ training (e.g., `solver='liblinear'`, `multi_class='ovr'`)
- Hyperparameters control the learning process and model behavior

### Classification Algorithms

**Logistic Regression**

- Despite the name, it's a _classification_ algorithm (not regression)
- Models probability that an example belongs to each class
- Originally designed for binary classification; extended to multiclass via schemes

**Support Vector Machines (SVM / SVC)**

- Finds optimal decision boundaries (hyperplanes) that separate classes
- `SVC` = Support Vector Classifier (scikit-learn class)
- Works well for high-dimensional data

**Decision Trees**

- Tree-like model of decisions based on feature values
- Easy to interpret; can overfit if not pruned
- Foundation for ensemble methods

**Ensemble Methods**

- Combine multiple models to improve performance
- Examples: Random Forest, AdaBoost, Gradient Boosting
- Voting Classifier: combines predictions from multiple different algorithms

**K-Nearest Neighbors (KNeighbors)**

- Classifies based on the majority class of K nearest training examples
- Simple but can be slow on large datasets

**Naive Bayes (GaussianNB)**

- Probabilistic classifier based on Bayes' theorem
- Assumes features are independent (the "naive" assumption)

**Neural Networks (MLP)**

- Multi-Layer Perceptron: layers of interconnected nodes
- Can learn complex nonlinear patterns
- Requires more data and computational resources

### Multiclass Classification Strategies

**One-vs-Rest (OvR) / One-vs-All**

- Trains one binary classifier per class
- Each classifier answers: "Is this example class A or not-A?"
- Final prediction: class with highest confidence
- Set with `multi_class='ovr'`

**Multinomial**

- Single model that directly handles multiple classes
- Uses cross-entropy loss to optimize probabilities across all classes
- Requires specific solvers: `lbfgs`, `sag`, `saga`, `newton-cg`
- Set with `multi_class='multinomial'`

**Two-class Classifier**

- Binary classification: only 2 possible classes
- Not applicable to the cuisine problem (5 classes)

### Optimization & Solvers

**Solver**

- The optimization algorithm used to find the best model parameters
- Different solvers work better for different data structures and problem sizes
- Common solvers for LogisticRegression:
  - `liblinear`: Good for small datasets; supports OvR
  - `lbfgs`: Default; works for multinomial; handles large datasets
  - `sag` / `saga`: Stochastic Average Gradient; fast on large datasets
  - `newton-cg`: Newton's method; good for multinomial

**Cross-Entropy Loss**

- Loss function measuring difference between predicted probabilities and true labels
- Used when `multi_class='multinomial'`
- Lower loss = better model fit

**Optimization Problem**

- Finding parameter values that minimize the loss function
- The solver's job is to solve this optimization problem efficiently


### Advanced Concepts

**AutoML (Automated Machine Learning)**

- Automatically tries multiple algorithms and hyperparameters
- Selects the best model based on performance metrics
- Saves time compared to manual experimentation

**Nonparametric Methods**

- Models whose complexity grows with the data
- Don't assume a fixed functional form (like linear models do)
- Examples: Decision Trees, KNN, kernel methods
- More flexible but can overfit

