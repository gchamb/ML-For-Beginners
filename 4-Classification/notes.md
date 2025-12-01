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
