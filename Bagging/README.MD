## @Author : Saurabh

# *Bagging Technique* :

Bagging is combining the results of multiple models (for instance, all decision trees) to get a generalized result. 
We generally devide the dataset into sub-dataset and test those sun data on base models.
Bootstrapping is a sampling technique in which we create subsets of observations from the original dataset, with replacement. 
The size of the subsets is the same as the size of the original set.

ex-
Original data -
              - sub_data1
              - sub_data2
              - sub_data3
              - sub_data4

- Multiple subsets are created from the original dataset, selecting observations with replacement.
- A base model (weak model) is created on each of these subsets.
- The models run in parallel and are independent of each other.
- The final predictions are determined by combining the predictions from all the models


> # 1. Bagging meta estimator 

Bagging meta-estimator is an ensembling algorithm that can be used for both classification (BaggingClassifier) and regression (BaggingRegressor) problems. 
It follows the typical bagging technique to make predictions. Following are the steps for the bagging meta-estimator algorithm:

Random subsets are created from the original dataset (Bootstrapping).

The subset of the dataset includes all features.

A user-specified base estimator is fitted on each of these smaller sets.

Predictions from each model are combined to get the final result. 

- Synatax:

> from sklearn.ensemble import BaggingClassifier
> from sklearn import tree 
> model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
> model.fit(x_train, y_train)
> model.score(x_test,y_test)

> from sklearn.ensemble import BaggingRegressor
> model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
> model.fit(x_train, y_train)
> model.score(x_test,y_test)

## Parameters :

### base_estimator:
It defines the base estimator to fit on random subsets of the dataset.
When nothing is specified, the base estimator is a decision tree.

### n_estimators:
It is the number of base estimators to be created.
The number of estimators should be carefully tuned as a large number would take a very long time to run, while a very small number might not provide the best results.

### max_samples:
This parameter controls the size of the subsets.
It is the maximum number of samples to train each base estimator.

### max_features:
Controls the number of features to draw from the whole dataset.
It defines the maximum number of features required to train each base estimator.

### n_jobs:
The number of jobs to run in parallel.
Set this value equal to the cores in your system.
If -1, the number of jobs is set to the number of cores.

### random_state:
It specifies the method of random split. When random state value is same for two models, the random selection is same for both models.
This parameter is useful when you want to compare different models.


> # 2. Random Forest 

It is an extension of the bagging estimator algorithm. The base estimators in random forest are decision trees. 
Unlike bagging meta estimator, random forest randomly selects a set of features which are used to decide the best split at each node of the decision tree.
 
Random subsets are created from the original dataset (bootstrapping).
At each node in the decision tree, only a random set of features are considered to decide the best split.
A decision tree model is fitted on each of the subsets.
The final prediction is calculated by averaging the predictions from all decision trees.

## Parameters :

### n_estimators:
It defines the number of decision trees to be created in a random forest.
Generally, a higher number makes the predictions stronger and more stable, but a very large number can result in higher training time.

### criterion:
It defines the function that is to be used for splitting.
The function measures the quality of a split for each feature and chooses the best split.

### max_features :
It defines the maximum number of features allowed for the split in each decision tree.
Increasing max features usually improve performance but a very high number can decrease the diversity of each tree.

### max_depth:
Random forest has multiple decision trees. This parameter defines the maximum depth of the trees.

### min_samples_split:
Used to define the minimum number of samples required in a leaf node before a split is attempted.
If the number of samples is less than the required number, the node is not split.

### min_samples_leaf:
This defines the minimum number of samples required to be at a leaf node.
Smaller leaf size makes the model more prone to capturing noise in train data.

### max_leaf_nodes:
This parameter specifies the maximum number of leaf nodes for each tree.
The tree stops splitting when the number of leaf nodes becomes equal to the max leaf node.

### n_jobs:
This indicates the number of jobs to run in parallel.
Set value to -1 if you want it to run on all cores in the system.

### random_state:
This parameter is used to define the random selection.
It is used for comparison between various models.
