## Notes:

`feature` - The input/s of the model and typically the column heads of your data. A measurable property.

> Ex: The color, size, weight

`target` - The output/s fo the model and the answer of the question you are asking.

> Ex. The **price** of a given neighborhood during winter

### Choosing Features

`feature selection` - A subset of the original features

`feature extraction` - derived from the original features

### Splitting Data

`Training` - Training data should take up majority of the split and used to actually train the model.

`Testing` - Testing data should be the portion of the data that is used to actually test your model

`Validation` - Validation data should be the portion of the data that is used to tune your model further based on the results of your testing.

### Model Fitting

This is the process where we place our features (aka our inputs) and our target (aka our output) into our model.

`underfitting` - The data cannot predict anything from the training data nor the unseen data.

`overfitting` - The model is too closely aligned with the training data causing the predictions with the unseen data to be off.

![overfitting model](images/overfitting.png)

> Infographic by [Jen Looper](https://twitter.com/jenlooper)

### Tuning

`Hyperparameter Tuning` - is the process of finding the hyperparameter configuration that yields the best performance. This process is often computationally expensive and manual.
