import pandas as pd

train_data = pd.read_csv('train.csv')

X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

import pandas as pd

# explore missing data
print(X.isnull().sum())

# Handle the missing data
X.dropna(subset=['Embarked'], inplace=True)
y = y[X.index]
X.drop(['Cabin', 'Age', 'Name', 'Ticket'], axis=1, inplace=True)

# convert categorical features to numerical features
X['Sex'] = X['Sex'].map({'male' : 0, 'female' : 1})
X['Embarked'] = X['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2})

# split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# train model
# from sklearn.ensemble import RandomForestClassifier

# # use grid search to find the best parameters
# from sklearn.model_selection import GridSearchCV

# param_grid = {'n_estimators' : [100, 200, 300, 400, 500],
#                 'max_depth' : [1, 2, 3, 4, 5]}
# grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
# grid.fit(X_train, y_train)
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_)
# print(grid.score(X_test, y_test))

# # extract the best model
# rf = grid.best_estimator_

# # visualize how important each feature is
# import matplotlib.pyplot as plt

# plt.barh(X.columns, rf.feature_importances_)
# plt.show()

male_survivors = train_data[(train_data['Survived']==True) & (train_data['Sex'] == 'male')]
print(male_survivors.shape[0])