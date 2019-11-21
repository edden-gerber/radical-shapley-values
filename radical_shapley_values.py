import numpy as np
import pandas as pd

#%% Shapley function:
def compute_shapley_values(payoff_function, X, y=None, zero_payoff=0):

    # Nested functions
    def get_coalition(num_features, comb_index):
        combination = np.zeros(num_features)
        temp = list(map(int,list(bin(comb_index)[2:]))) # change from comb_index+1
        combination[(num_features-len(temp)):] = temp
        return combination.astype('bool')

    def compute_shapley():
        shapley_values = dict()

        for player in range(0,num_players):
            not_participating = participation[player]==False
            indexes_not_participating = np.where(not_participating)[0]
            indexes_participating = indexes_not_participating + 2**(num_players - player - 1)
            non_part_coalition_size = coalition_size[indexes_not_participating]

            player_shap_value = 0
            for i in range(0,len(indexes_not_participating)):
                marginal_payoff = payoff[indexes_participating[i]] - payoff[indexes_not_participating[i]]
                S = non_part_coalition_size[i]
                player_shap_value += np.math.factorial(S) * np.math.factorial(num_players-S-1) / np.math.factorial(num_players) * marginal_payoff

            shapley_values[player] = player_shap_value

        return shapley_values


    num_players = X.shape[1]
    num_coalitions = 2**(num_players) # removed -1

    participation = np.ones([num_players, num_coalitions]).astype('bool')
    payoff = dict()

    for coalition_index in range(0, num_coalitions):
        coalition_players = get_coalition(num_players, coalition_index)
        partial_X = X[X.columns[coalition_players]]
        participation[:,coalition_index] = coalition_players
#        payoff = coalition_size
        if all(~participation[:,coalition_index]): # if there are no players in this coalition, the payoff function is not necessarily defined (e.g. train a model with no data)
            payoff[coalition_index] = zero_payoff
        else:
            if y is None:
                payoff[coalition_index] = payoff_function(partial_X)
            else:
                payoff[coalition_index] = payoff_function(partial_X, y)
            
    coalition_size = np.sum(participation.astype('int'),axis=0)
    shapley_values = compute_shapley()

#    return shapley_values, payoff, participation
    return shapley_values

#%% Rehsape output - use to convert dictionary output to array
def reshape_shapley_output(shapley_dict):
    
    if type(shapley_dict[0]).__name__=='float':
        shapley_array = np.zeros([1, len(shapley_dict)])
    else:
        shapley_array = np.zeros([len(shapley_dict[0]), len(shapley_dict)])
    for i in range(0,len(shapley_dict)):
        shapley_array[:,i] = shapley_dict[i]
#        shapley_array[:,i] = np.array(list(shapley_dict.items())[0][i])
    
    return shapley_array

#%% Example payoff functions

# XGBoost regression
import xgboost as xgb
def shap_payoff_XGBreg(X,y):
    xgb_model = xgb.XGBRegressor(random_state=1)
    xgb_model.fit(X, y)
    out = xgb_model.predict(X)
    return out


# XGBoost classification
def shap_payoff_XGBclas(X,y):
    xgb_model = xgb.XGBClassifier(random_state = 1, objective='binary:logistic')
    xgb_model.fit(X, y)
#    out = xgb_model.predict_proba(X)[:,1]
    out = xgb_model.predict(X, output_margin=True)
    return out


# Random Forest
from sklearn.ensemble import RandomForestRegressor
def shap_payoff_random_forest(X,y):
    forest = RandomForestRegressor(random_state=1)
    forest.fit(X,y)
    out = forest.predict(X)
    return out


# Decision tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
def shap_payoff_tree(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    tree = DecisionTreeRegressor(random_state=1)
    tree.fit(X_train,y_train)
    return tree.predict(X_valid)


# Isolation forest
from sklearn.ensemble import IsolationForest
def shap_payoff_isolation_forest(X):
    model = IsolationForest(behaviour = 'new', random_state = 1)
    model.fit(X)
    anomaly_scores = model.decision_function(X)
    return anomaly_scores


# KNN
from sklearn.neighbors import KNeighborsClassifier
def shap_payoff_KNN(X, y):
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    return knn.predict_proba(X)[:,1]
