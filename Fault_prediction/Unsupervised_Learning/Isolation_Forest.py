import numpy as np
# import eif as iso
import pandas as pd
from Simulation.Parameters import SET_PARAMS
from Fault_prediction.Fault_utils import Dataset_order
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

np.random.seed(42)

def getDepth(x, root, d):
    n = root.n
    p = root.p
    if root.ntype == 'exNode':
        return d
    else:
        if (x-p).dot(n) < 0:
            return getDepth(x,root.left,d+1)
        else:
            return getDepth(x,root.right,d+1)
        
def getVals(forest,x,sorted=True):
    theta = np.linspace(0,2*np.pi, forest.ntrees)
    r = []
    for i in range(forest.ntrees):
        temp = forest.compute_paths_single_tree(np.array([x]),i)
        r.append(temp[0])
    if sorted:
        r = np.sort(np.array(r))
    return r, theta

def IsoForest(path, featureExtractionMethod, treeDepth, multi_class = False, constellation = False, lowPredictionAccuracy = False, MovingAverage = True, includeAngularMomemntumSensors = False, includeModelled = False):
    X_list = []
    Y_list = []

    pathFiles = SET_PARAMS.path

    buffer = False
    SET_PARAMS.buffer_size = 2

    if constellation:
        for satNum in range(SET_PARAMS.Number_of_satellites):
            print(satNum)
            SET_PARAMS.path = pathFiles + str(satNum) + "/"
            for index in range(SET_PARAMS.number_of_faults):
                name = SET_PARAMS.Fault_names_values[index+1]
                if multi_class:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, constellation = constellation, multi_class = True, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors, includeModelled = includeModelled)
                else:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, constellation = constellation, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors, includeModelled = includeModelled)
                X_list.append(X)    
                Y_list.append(Y)

    else:
        for index in range(SET_PARAMS.number_of_faults):
            name = SET_PARAMS.Fault_names_values[index+1]
            if multi_class:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors, includeModelled = includeModelled)
            else:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors, includeModelled = includeModelled)

            print(ColumnNames)
            print(X.shape)
            if SET_PARAMS.Fault_names_values[index+1] != "None":
                mask = (Y[:, 0] == 1)
                Y = Y[mask]
                X = X[mask]

                print(X.shape)

                numberOfRows = X.shape[0]
                randomIndices = np.random.choice(numberOfRows, size = 100, replace = False)
                X = X[randomIndices, :]
                Y = Y[randomIndices, :]

                print(Y.shape)

   
            X_list.append(X)    
            Y_list.append(Y)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)

    print(X.shape)

    random_state = np.random.RandomState(42)

    Y = Y *-2

    Y = Y + 1

    for contamination in range(1, 20):

        print(contamination)

        model=IsolationForest(n_estimators=100,max_samples='auto',contamination=contamination/100,random_state=random_state, bootstrap = True)

        model.fit(X)

        scores = model.decision_function(X)

        anomaly_score = model.predict(X)



        # step = []

        # initVal = Y[0]

        # for val in Y:
        #     if val != initVal:
        #         step.append(-1)
        #     else:
        #         step.append(1)
        #     initVal = val

        # step = np.array(step)

        accuracy = 100*list(anomaly_score).count(-1)/(np.count_nonzero(Y==-1))
        print("Accuracy of the model:", accuracy)

        cm = confusion_matrix(anomaly_score, Y)

        print(cm)

        # cm = confusion_matrix(anomaly_score, step)

        # print(cm)

        # print(list(anomaly_score).count(-1))

        # fromIndex = 250000

        # plt.figure()
        # plt.plot(range(len(Y[fromIndex:])), Y[fromIndex:], 'b', alpha = 0.3)
        # plt.plot(range(len(anomaly_score[fromIndex:])), anomaly_score[fromIndex:], 'r', alpha = 0.3)
        # plt.show()

        pickle.dump(model, open(path + '/IsolationForest' + str(contamination/100) + '.sav', 'wb'))

    