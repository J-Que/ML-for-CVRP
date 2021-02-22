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
from sklearn.neural_network import MLPClassifier

path = 'C:/Users/queve/Desktop/CVRP'

def get_data(n):
    # read the csv file
    df = pd.read_csv(path + '/CVRP_dataset.csv').sample(n=n, random_state=1)
    data = df.copy()
    
    # dropped the unwanted columns
    columns = ['ID','Instance', 'Inst_Type', 'DptModule', 'CtyModule', 'DmdModule', 'Label', 'L1.CWSoln', 'L2.SPSoln', 'L3.GASoln', 'L4.SOMSoln']
    data.drop(axis=1, columns=columns, inplace=True)

    # conduct the split of training and test data
    return train_test_split(data, df['Label'], test_size=0.2, random_state=1)

def get_non_pca_parameters(max_iter, alpha, power_t, layers, solvers, learning_rates, learning_rates_inits, X, y):
    # build hyperparameters list for non pca classifiers
    parameters  = []
    for i in alpha:
        for j in layers:
            for k in solvers:
                if k == 'sgd':
                    for l in max_iter:
                        for m in learning_rates_inits:
                            for n in learning_rates:
                                if n == 'invscaling':
                                    for o in power_t:
                                        parameters.append(['None', l, i, o, j, k, n, m, X, y])
                                else:
                                    parameters.append(['None', l, i, 0.5, j, k, n, m, X, y])
                if k == 'adam':
                    for l in max_iter:
                        for m in learning_rates_inits:
                            parameters.append(['None', l, i, 0.5, j, k, 'constant', m, X, y])
                else:
                    parameters.append(['None', 200, i, 0.5, j, k, 'constant', 0.001, X, y])
    return parameters

def get_pca_parameters(max_iter, alpha, power_t, layers, solvers, learning_rates, learning_rates_inits, pc_range, X, y):
    # build hyperparameters list for pca classifiers
    parameters  = []
    for i in alpha:
        for j in layers:
            for k in solvers:
                if k == 'sgd':
                    for l in max_iter:
                        for m in learning_rates_inits:
                            for n in learning_rates:
                                if n == 'invscaling':
                                    for o in power_t:
                                        for p in pc_range:
                                            X_copy = X.iloc[:, :p]
                                            parameters.append([p, l, i, o, j, k, n, m, X_copy, y])
                                else:
                                    for p in pc_range:
                                        X_copy = X.iloc[:, :p]
                                        parameters.append([p, l, i, 0.5, j, k, n, m, X_copy, y])
                if k == 'adam':
                    for l in max_iter:
                        for m in learning_rates_inits:
                            for p in pc_range:
                                X_copy = X.iloc[:, :p]
                                parameters.append([p, l, i, 0.5, j, k, 'constant', m, X_copy, y])
                else:
                    for p in pc_range:
                        X_copy = X.iloc[:, :p]
                        parameters.append([p, 200, i, 0.5, j, k, 'constant', 0.001, X_copy, y])

    return parameters

def mlp_model(param):
    # get the parameters from the param list
    pc = param[0]
    max_iter = param[1]
    alpha = param[2]
    power_t = param[3]
    layers = param[4]
    solver = param[5]
    learning_rate = param[6]
    learning_rate_init = param[7]
    x, y = param[8], param[9]
    
    #Create a single classifer based off the given parameters and get its execution time
    start = time.time()
    clf = MLPClassifier(max_iter=max_iter, alpha=alpha, power_t=power_t, hidden_layer_sizes=layers, solver=solver, learning_rate=learning_rate, learning_rate_init=learning_rate_init)
    clf.fit(x, y)
    scores = cross_val_score(clf, x, y, cv=5)
    total_time = time.time() - start

    return pd.DataFrame({'PCA':pc, 'Max iter':max_iter, 'Alpha':str(alpha), 'Power t':str(power_t),'Layers':str(layers), 'Solver':solver, 'Learning Rate':learning_rate, 'Learning Rate Init':learning_rate_init, 'Mean CV Score':np.average(scores), 'STD CV Score':np.std(scores), 'Execution Time':total_time}, index=[0])

def create_models_parallel(parameters, file_name):
    #create a dataframe to hold the results
    model_descriptions = pd.DataFrame(columns=['PCA', 'Max iter', 'Alpha', 'Power t', 'Layers','Solver', 'Learning Rate', 'Learning Rate Init', 'Mean CV Score', 'STD CV Score', 'Execution Time'])
    count = 1

    # create all the classifiers in parallel and record the results{}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        models = [executor.submit(mlp_model, param) for param in parameters]
        for run in concurrent.futures.as_completed(models):
            model_descriptions = model_descriptions.append(run.result(), ignore_index=True)
            model_descriptions.to_csv(file_name)
            print('Run', count, 'of', len(parameters), '\n', run.result())
            count += 1
    
def main(n):
    #Parameter elements
    solvers = ['sgd', 'adam', 'lbfgs']
    max_iter =[5000, 6000]
    power_t = [0.5]
    learning_rates = ['constant', 'invscaling', 'adaptive']
    alpha = [0.00001,  0.00005, 0.0001, 0.0005, 0.001]
    layers = [(24,), (21,),(18,),(15,),(12,),(9,),(6,),(3,),(27,)]
    learning_rates_inits = [0.0001, 0.0005, 0.001, 0.005, 0.01] 
    pc_range = range(23, 0, -1)
        
    # Get and split the data
    X_train, X_test, y_train, y_test = get_data(n)

    #scale the data
    scaler = MinMaxScaler(feature_range=(-1,1), copy=False)
    X_train = scaler.fit_transform(X_train)

    # convert back into dataframe
    X_train = pd.DataFrame(X_train, columns=range(1,24))
    
    #Get combinations of parameters without PCA
    parameters = get_non_pca_parameters(max_iter, alpha, power_t, layers, solvers, learning_rates, learning_rates_inits, X_train, y_train)
    
    #Conduct PCA on data set
    pca = PCA()
    X_train = pca.fit_transform(X_train)

    # Convert data back into dataframe after transformations
    X_train = pd.DataFrame(X_train, columns=range(1,24))

    #Get combinations of parameters with PCA
    parameters = parameters + get_pca_parameters(max_iter, alpha, power_t, layers, solvers, learning_rates, learning_rates_inits, pc_range, X_train, y_train)
    
    random.shuffle(parameters)
    print(len(parameters))
    #Create different models based on parameters all in parallel (Grid Search)
    create_models_parallel(parameters,  path + '/model descriptions/mlp_model_descriptions_%s.csv'%n)
    
if __name__ == '__main__':
    start = time.time()
    main(4897)
    print('\n\n---', ' MLP TOTAL EXECUTION TIME: ', time.time() - start, '---\n\n')