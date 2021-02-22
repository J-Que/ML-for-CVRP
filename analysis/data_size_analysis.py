import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, scale, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, plot_confusion_matrix, f1_score, precision_score, recall_score
#print shape of adjusted dataframe
path = 'C:/Users/jdque/Desktop/CVRP/'

df = pd.read_csv(path + 'CVRP_dataset.csv')
labels = df['Label']

columns=['ID','Instance', 'Inst_Type', 'DptModule', 'CtyModule', 'DmdModule', 'Label']
df.drop(axis=1, columns=columns,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=1)

df = pd.concat([X_train, y_train], axis=1) 
testing = pd.concat([X_test, y_test], axis=1)
costs = testing[['L1.CWSoln', 'L2.SPSoln', 'L3.GASoln', 'L4.SOMSoln', 'Label']]

df.drop(axis=1, columns=['L1.CWSoln', 'L2.SPSoln', 'L3.GASoln', 'L4.SOMSoln'],inplace=True)
X_test.drop(axis=1, columns=['L1.CWSoln', 'L2.SPSoln', 'L3.GASoln', 'L4.SOMSoln'],inplace=True)

def get_parameters():
    sizes = [i for i in range(10, 3911, 10)] + [3917]
    learners = ['knn', 'mlp', 'rf']
    param = []
    for i in sizes:
        for j in learners:
            param.append([i,j])
    return param

def pipe(clf):







    # create the scaling and pca objects
    scaler = MinMaxScaler(feature_range=(-1,1), copy=False)
    pca = PCA(n_components=4)

    # create the classifiers
    if clf == 'knn':
        knn_clf = KNeighborsClassifier(n_neighbors=36, weights='distance', p=2)
        return Pipeline([('scaler', scaler), ('pca', pca), ('knn', knn_clf)])
    elif clf == 'rf':
        rf_clf = RandomForestClassifier(criterion='gini', max_depth=24, n_estimators=160, random_state=1)
        return Pipeline([('rf', rf_clf)])
    elif clf == 'mlp':
        #mlp_clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, solver='adam', learning_rate='constant', learning_rate_init=0.005, alpha=0.05, random_state=1)
        mlp_clf = MLPClassifier(hidden_layer_sizes=(24,), max_iter=1000, solver='lbfgs', learning_rate='constant', learning_rate_init=0.001, alpha=0.9, random_state=1)
        #mlp_clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, solver='sgd', learning_rate='adaptive', learning_rate_init=0.01, alpha=0.0001, random_state=1)
        return Pipeline([('scaler', scaler), ('pca', pca), ('mlp', mlp_clf)])
    
def get_data(n):
    # read in data and split
    X_train = df.sample(n=n, random_state=1)
    y_train = X_train['Label']
    X_train.drop(axis=1, columns=['Label'],inplace=True)
    
    return X_train, y_train

def pred_cost(row):
    if row['Prediction'] == 'CW':
        return row['L1.CWSoln']
    elif row['Prediction'] == 'SP':
        return row['L2.SPSoln']
    elif row['Prediction'] == 'GA':
        return row['L3.GASoln']
    elif row['Prediction'] == 'SOM':
        return row['L4.SOMSoln']
    
def savings(prediction):
    temp_costs = costs.copy()
    temp_costs['Prediction'] = prediction
    mlCosts = []
    for index, row in costs.iterrows():
        mlCosts.append(pred_cost(row))
    
    return np.average(mlCosts)

def runs(param):
    n, clf_name = param[0], param[1]
    clf = pipe(clf_name)
    X_train, y_train,  = get_data(n)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    pred = clf.predict(X_test)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall(y_test, pred)

    mlCosts  = savings(pred, costs)

    return pd.DataFrame({'Classifier': clf_name, 'Data Size':n, 'Accuracy':score, 'Precision':precision, 'Recall':recall, 'F1-Score':f1, 'ML Cost':mlCosts})

def main():
    parameters = get_parameters()

    #create a dataframe to hold the results
    model_descriptions = pd.DataFrame(columns=['Classifier', 'Data Size', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ML Costs'])
    count = 1

    # create all the classifiers in parallel and record the results{}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        models = [executor.submit(runs, param) for param in parameters]
        for run in concurrent.futures.as_completed(models):
            model_descriptions = model_descriptions.append(run.result(), ignore_index=True)
            model_descriptions.to_csv(path + '/model descriptions/size_descriptions.csv')
            print('Run', count, 'of', len(parameters), '\n', run.result())
            count += 1

if __name__ == '__main__':
    main()