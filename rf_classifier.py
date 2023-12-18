from sklearn.ensemble import RandomForestClassifier
import torch, numpy as np, json, pickle, time, os, sys
from tqdm import tqdm
from metrics import get_metrics_classicalml
from data_creator import load_train_test_data, get_ae_data, get_lda_data, get_pca_data
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def train_rf(dim_reduction):
    
    print(f"Training rf with {dim_reduction}")
    print(f"Generating data with {dim_reduction}")
    data_folder= os.getcwd() + "/att_faces"
    train_data, train_label, val_data, test_label = load_train_test_data(data_folder)
    train_arr =None
    test_data_arr=None
    if(dim_reduction=="ae"):
        train_arr = get_ae_data(train_data,'recognition')
        test_data_arr =   get_ae_data(val_data,'recognition')
    elif(dim_reduction=="pca"):
        train_arr, test_data_arr = get_pca_data(train_data,val_data)
    elif(dim_reduction=="lda"):
        train_arr,test_data_arr = get_lda_data(train_data,val_data)


    print("Train Shape - ", train_arr.shape)
    print("Test Shape - ", train_arr.shape)


    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    rf = RandomForestClassifier(100,n_jobs=-1, random_state=42)
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    start_time = time.time()
    grid_search.fit(train_arr,train_label)
    print("GS Time - ", time.time()-start_time)
    print("Best Params - ", grid_search.best_params_)

    random_forest = RandomForestClassifier(grid_search.best_params_['max_depth'], grid_search.best_params_['max_features'], grid_search.best_params_['min_samples_leaf'],grid_search.best_params_['min_samples_split'],grid_search.best_params_['n_estimators'])
    random_forest.fit(train_arr,train_label)

    print('Testing...')
    

    print("Label shape - ", test_label.shape)
    print("Input shape - ", test_data_arr.shape)

    pred_label = random_forest.predict(test_data_arr)

    ac,pr,re,f1 = get_metrics_classicalml(test_label, pred_label)
    cm = confusion_matrix(test_label, pred_label)

    print("Accuracy - ",round(ac,3))
    print("Precision - ",round(pr,3))
    print("Recall - ",round(re,3))
    print("F-1 - ",round(f1,3))

    print(classification_report(test_label,pred_label))
    print("Confusion Matrix:")
    print(cm)






if __name__=="__main__":


    train_rf("ae")
    train_rf("pca")
    train_rf("lda")



