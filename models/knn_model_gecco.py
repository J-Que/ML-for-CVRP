import os
import time
import numpy as np
import pandas as pd
import random
import concurrent.futures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

path = 'C:/Users/queve/Desktop/CVRP'

def get_data(n):
    # read the csv file
    df = pd.read_csv(path + '/CVRP_dataset.csv').sample(n=n, random_state=1)
    data = df.copy()

    # dropped the unwanted columns
    columns = ['ID','Instance', 'Inst_Type', 'DptModule', 'CtyModule', 'DmdModule', 'Label', 'L1.CWSoln', 'L2.SPSoln', 'L3.GASoln', 'L4.SOMSoln' ]
    data.drop(axis=1, columns=columns, inplace=True)

    # conduct the split of training and test data
    return train_test_split(data, df['Label'], test_size=0.2, random_state=1)

def get_non_pca_parameters(weights, neighbors, p_int, X, y):
    # build hyperparameters list for non pca classifiers
    parameters  = []
    for i in weights:
        for j in neighbors:
            for k in p_int:
                parameters.append([i, j, k, 'None', X, y])
    return parameters

def get_pca_parameters(weights, neighbors, p_int, pc_range, X, y):
    # build hyperparameters list for pca classifiers
    parameters  = []
    for i in weights:
        for j in neighbors:
            for k in p_int:
                for l in pc_range:
                    X_copy = X.iloc[:, :l]
                    parameters.append([i, j, k, l, X_copy, y])               
    return parameters

def knn_model(param):
    # get the parameters from the param list
    weights = param[0]
    neighbors = param[1]
    p_int = param[2]
    pc = param[3]
    x, y = param[4], param[5]
    
    #Create a single classifer based off the given parameters and get its execution time
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=neighbors, weights=weights, p=p_int)
    scores = cross_val_score(clf, x, y, cv=LeaveOneOut())
    total_time = time.time() - start
    
    return pd.DataFrame({'PCA':pc, 'Neighbors':neighbors, 'Weights':weights, 'P Integer':p_int, 'Mean CV Score':np.average(scores), 'STD CV Score':np.std(scores), 'Execution Time':total_time}, index=[0])

def create_models_parallel(parameters, file_name):
    #create a dataframe to hold the results
    model_descriptions = pd.DataFrame(columns=['PCA', 'Neighbors', 'Weights', 'P Integer', 'Mean CV Score', 'STD CV Score', 'Execution Time'])
    count = 1

    # create all the classifiers in parallel and record the results{}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        models = [executor.submit(knn_model, param) for param in parameters]
        for run in concurrent.futures.as_completed(models):
            model_descriptions = model_descriptions.append(run.result(), ignore_index=True)
            model_descriptions.to_csv(file_name)
            print('Run', count, 'of', len(parameters), '\n', run.result())
            count += 1
    
def main(n):
    #Parameter elements 
    weights = ['distance', 'uniform']
    neighbors = range(1,200,2)
    p_int = [1, 2]
    pc_range = range(23, 0, -1)
    
    # Get and split the data
    X_train, X_test, y_train, y_test = get_data(n)
    
    #scale the data (requiered for knn and pca)
    scaler = MinMaxScaler(feature_range=(-1,1), copy=False)
    X_train = scaler.fit_transform(X_train)

    # Convert data back into dataframe
    X_train = pd.DataFrame(X_train, columns=range(1,24))

    #Get combinations of parameters without PCA
    parameters = get_non_pca_parameters(weights, neighbors, p_int, X_train, y_train)
    
    #Conduct PCA on data set
    pca = PCA()
    X_train = pca.fit_transform(X_train)

    # Convert data back into dataframe after transformations
    X_train = pd.DataFrame(X_train, columns=range(1,24))

    #Get combinations of parameters with PCA
    parameters = parameters + get_pca_parameters(weights, neighbors, p_int, pc_range, X_train, y_train)

    random.shuffle(parameters)
    print(len(parameters))
    #Create different models based on parameters all in parallel (Grid Search)
    create_models_parallel(parameters,  path + '/model descriptions/knn_model_descriptions_%s.csv'%n )
    

if __name__ == '__main__':
    overall = time.time()
    start = time.time()
    main(4897)
    print('\n\n---', ' KNN TOTAL EXECUTION TIME: ', time.time() - start, '---\n\n')

    import rf_model_gecco as rf
    start = time.time()
    rf.main(4897)
    print('\n\n---', ' KNN TOTAL EXECUTION TIME: ', time.time() - start, '---\n\n')

    import mlp_model_gecco as mlp
    start = time.time()
    mlp.main(4897)
    print('\n\n---', ' KNN TOTAL EXECUTION TIME: ', time.time() - start, '---\n\n')