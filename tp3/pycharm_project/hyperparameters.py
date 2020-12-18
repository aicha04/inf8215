from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

# Function used to get importance of features
def get_feature_importances(feature_list, importances):
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                            zip(feature_list, importances)]

    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot(111)
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation = 'vertical', color='#eb483c')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    fig.tight_layout()

    now = datetime.datetime.now()
    current_time = now.strftime("%H_%M_%S")
    file_path = 'importance_' + current_time + '.png'
    plt.savefig(file_path)
    plt.close(fig)

# Function used to train model with random forest regressor
def train_model(train_features, train_labels,random):
    rf = RandomForestClassifier(n_estimators=1000, random_state=random)

    rf.fit(train_features, train_labels)

    #joblib.dump(rf, 'model.sav')

    return rf

# Function used to train model with random forest regressor
def cross_validation(features, labels, random):
    rf = RandomForestClassifier(n_estimators=1000, random_state=random)
    
    rf.fit(features, labels)

    scores = cross_val_score(rf, features, labels, cv=5)
    print(scores)
    print("Mean 5-Fold R Squared: {}".format(np.mean(scores)))

    return (np.mean(scores),rf)

def create_grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    return random_grid

def grid_search():
    (X,Y,feature_list) = get_data()
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [False],
        'max_depth': [None],
        'max_features': [2, 3],
        'min_samples_leaf': [1,2,3],
        'min_samples_split': [2,3,4],
        'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    rf = RandomForestClassifier()# Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)

    # Fit the grid search to the data
    grid_search.fit(X,Y)
    print(grid_search.best_params_)

    best_grid = grid_search.best_estimator_
    scores = cross_val_score(best_grid, X, Y, cv=5)
    print(scores)
    print("Mean 5-Fold R Squared: {}".format(np.mean(scores)))

def find_hyperparameters():
    (X,Y,feature_list) = get_data()
    rf = RandomForestClassifier(n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False)
    rf.fit(X, Y)

    scores = cross_val_score(rf, X, Y, cv=5)
    print(scores)
    print("Mean 5-Fold R Squared: {}".format(np.mean(scores)))

#{'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
def random_search():
    (X,Y,feature_list) = get_data()
    random_grid = create_grid()
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
    rf_random.fit(X, Y)
    print(rf_random.best_params_)

    best_random = rf_random.best_estimator_
    scores = cross_val_score(best_random, X, Y, cv=5)
    print(scores)
    print("Mean 5-Fold R Squared: {}".format(np.mean(scores)))

def get_data():
    # Loading the data and checking its dimension
    data_train = pd.read_csv('data/train.csv', header=None, skiprows=1)

    # Separating the features and the true value(x,y)
    data_train = data_train.drop(0, axis=1)
    last_column = data_train.columns[-1]
    Y = data_train[last_column]
    X = data_train.drop(last_column, axis=1)
    feature_list = list(X.columns)
    numberFeature = X.shape[1]

    #normalisation of features
    print(X.shape)
    mean = X.mean()
    var = X.var()
    for i in range(1,numberFeature+1):
        if(var[i] !=0):
            X[i] = (X[i]-mean[i])/var[i]


    # Loading it into a numpy array
    X = X.to_numpy()
    Y = Y.to_numpy()
    Y = Y == "phishing"
    assert not np.any(np.isnan(X))

    return (X,Y,feature_list)

def predict_usage():
    (X,Y,feature_list) = get_data()
    # X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.10)
    # print('Training Features Shape:', X_train.shape)
    # print('Training Labels Shape:', Y_train.shape)
    # print('Validating Features Shape:', X_validation.shape)
    # print('Validating Labels Shape:', Y_validation.shape)
    
    #(score,rf) = cross_validation(X,Y,42)
    # number = [3, 17, 24, 39, 55]
    # scores = []
    # for i in number:
    #     scores.append(cross_validation(X,Y,i))
    
    # # find best model
    # rf = train_model(X,Y,scores.index(max(scores)))
    # print(max(scores))

    # # predict validation set
    # predictions = rf.predict(X_validation)
    # Y_pred = (predictions > 0.5)
    # print(Y_pred)
    # df = pd.DataFrame(columns=['idx', 'status'])
    # i = 0
    # for pred in Y_pred:
    #     if pred:
    #         df = df.append({"idx": i, "status": "phishing"}, ignore_index=True)
    #     else:
    #         df = df.append({"idx": i, "status": "legitimate"}, ignore_index=True)
    #     i += 1
    # df.to_csv("data/predictions.csv", index=False)

    # get_feature_importances(feature_list, list(rf.feature_importances_))

    # errors = Y_pred == Y_validation

    # accuracy = np.sum(errors)/errors.shape[0]
    # print('Accuracy validation:', round(accuracy, 5), '%.')

    # Test data
    data_test = pd.read_csv('data/test.csv', header=None, skiprows=1)
    data_test = data_test.drop(0, axis=1)
    X_test = data_test

    #normalisation of features
    mean_test = X_test.mean()
    var_test = X_test.var()
    for i in range(1,numberFeature+1):
        if(var_test[i] !=0):
            X_test[i] = (X_test[i]-mean_test[i])/var_test[i]

    # Loading it into a numpy array
    X_test = X_test.to_numpy()

    # predict test set
    Y_pred = rf.predict(X_test)
    #Y_pred = (predictions > 0.5)
    df2 = pd.DataFrame(columns=['idx', 'status'])
    i = 0
    for pred in Y_pred:
        if pred:
            df2 = df2.append({"idx": i, "status": "phishing"}, ignore_index=True)
        else:
            df2 = df2.append({"idx": i, "status": "legitimate"}, ignore_index=True)
        i += 1
    df2.to_csv("data/submission.csv", index=False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plt.switch_backend('agg')
    grid_search()

