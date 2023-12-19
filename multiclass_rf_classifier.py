# Modified version of rf_classifier.py for 40-class classification task

from sklearn.ensemble import RandomForestClassifier
import torch, numpy as np, json, pickle, time, os, sys
from tqdm import tqdm
from metrics import get_metrics_classicalml
from data_creator import load_train_test_data, get_lda_data, get_pca_data
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from auto_encoder import get_ae_data




def train_rf(dim_reduction):
    
    print(f"Training rf with {dim_reduction}")
    print(f"Generating data with {dim_reduction}")
    data_folder= os.getcwd() + "/att_faces"
    train_data, train_label, val_data, test_label = load_train_test_data(data_folder,'identification')
    train_arr =None
    test_data_arr=None
    if(dim_reduction=="ae"):
        train_arr = get_ae_data(train_data,'identification')
        test_data_arr =   get_ae_data(val_data,'identification')
    elif(dim_reduction=="pca"):
        train_arr, test_data_arr = get_pca_data(train_data,val_data)
        train_label,test_label = torch.tensor(train_label),torch.tensor(test_label)
        train_arr,test_data_arr= torch.tensor(train_arr).to(torch.float32),torch.tensor(test_data_arr).to(torch.float32)
    elif(dim_reduction=="lda"):
        train_arr, test_data_arr = get_lda_data(train_data,train_label,val_data)
        train_label,test_label = torch.tensor(train_label),torch.tensor(test_label)
        train_arr,test_data_arr= torch.tensor(train_arr).to(torch.float32),torch.tensor(test_data_arr).to(torch.float32)

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
    grid_search.fit(train_arr.cpu(),train_label)
    print("GS Time - ", time.time()-start_time)
    print("Best Params - ", grid_search.best_params_)

    random_forest = RandomForestClassifier(max_depth=grid_search.best_params_['max_depth'], max_features=grid_search.best_params_['max_features'], min_samples_leaf=grid_search.best_params_['min_samples_leaf'],min_samples_split=grid_search.best_params_['min_samples_split'],n_estimators=grid_search.best_params_['n_estimators'],bootstrap=grid_search.best_params_['bootstrap'])
    random_forest.fit(train_arr.cpu(),train_label)

    print('Testing...')
    

    print("Label shape - ", test_label.shape)
    print("Input shape - ", test_data_arr.shape)

    pred_label = random_forest.predict(test_data_arr.cpu())

    ac,pr,re,f1 = get_metrics_classicalml(np.argmax(test_label,1), np.argmax(pred_label,1))
    cm = confusion_matrix(np.argmax(test_label,1), np.argmax(pred_label,1))

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



