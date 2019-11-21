# exact_shapely_values demo script

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import shap
import time
import matplotlib.pyplot as plt
import warnings

from radical_shapley_values import compute_shapley_values
from radical_shapley_values import shap_payoff_isolation_forest as shap_if
from radical_shapley_values import shap_payoff_XGB as shap_xgb
from radical_shapley_values import reshape_shapley_output


#%% Generate random data
num_samples = 10**5
num_features = 4

X = pd.DataFrame(np.random.randn(num_samples, num_features))
y = np.round(np.random.rand(num_samples))

#%% Run exact-shapley function for isolation forest

start = time.time()
shapley_values = reshape_shapley_output(compute_shapley_values(shap_if, X))
end = time.time()
print('Run time: %.3f' %(end-start))

#%% Run SHAP library for isolation forest

num_shap_samples = 1000 # number of samples for which to compute SHAP values. Total run time should be linear with this. 
num_samples_in_local_env = 100  # used to generate the local environment of each sample. 
                                # 100 is a standard parameter value, higher values lead to more exact shap values 
                                # with the cost of longer run times

model = IsolationForest(behaviour = 'new', random_state = 1)
model.fit(X)
explainer = shap.KernelExplainer(model.decision_function, X.iloc[0:num_samples_in_local_env])

start = time.time()
with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel_shap_values = explainer.shap_values(X.iloc[0:num_shap_samples])
end = time.time()
print('Run time: %.3f' %(end-start))


#%% Plot radical Shapley vs SHAP values: 
feature_idx = 0
plt.scatter(shapley_values[0:num_shap_samples,feature_idx], kernel_shap_values[0:num_shap_samples,feature_idx], s=5)
mn = np.min(kernel_shap_values[:,feature_idx])
mx = np.max(kernel_shap_values[:,feature_idx])
plt.plot([mn,mx],[mn,mx],'r--')
plt.xlabel('Radical Shapley values')
plt.ylabel('Kernel Explainer SHAP values')


#%% Run compute-shapley function for XGBoost
start = time.time()
shapley_values = reshape_shapley_output(compute_shapley_values(shap_xgb, X, y))
end = time.time()
print('Run time: %.3f' %(end-start))

#%% Run SHAP library for XGBoost
num_shap_samples = num_samples # number of samples for which to compute SHAP values. Total run time should be linear with this. 

xgb_model = xgb.XGBRegressor(scale_pos_weight=1, \
                          objective='binary:logistic', \
                          eval_metric='auc', \
                          max_depth=5, \
                          n_estimators=100)

xgb_model.fit(X, y)
explainer = shap.TreeExplainer(xgb_model)
start = time.time()
tree_shap_values = explainer.shap_values(X.iloc[0:num_shap_samples])
end = time.time()
print('Run time: %.3f' %(end-start))

#%% Plot radical Shapley vs SHAP values: 
feature_idx = 0
plt.scatter(shapley_values[0:num_shap_samples,feature_idx], tree_shap_values[0:num_shap_samples,feature_idx],s=5)
mn = np.min(tree_shap_values[:,feature_idx])
mx = np.max(tree_shap_values[:,feature_idx])
plt.plot([mn,mx],[mn,mx],'r--')
plt.xlabel('Radical Shapley values')
plt.ylabel('Tree Explainer SHAP values')
