from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from Fault_prediction.Fault_utils import Dataset_order
from sklearn.metrics import confusion_matrix
import numpy as np
from Simulation.Parameters import SET_PARAMS
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


def Random_Forest(path, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames):
    X_list = []
    Y_list = []

    pathFiles = SET_PARAMS.path

    buffer = False

    if multi_class:
        ignoreNormal = True
        startNum = 1
    else:
        ignoreNormal = False
        startNum = 0

    if (X == None).any() or (Y == None).any():
        if constellation:
            for satNum in range(SET_PARAMS.Number_of_satellites):
                print(satNum)
                SET_PARAMS.path = pathFiles + str(satNum) + "/"
                for index in range(startNum, SET_PARAMS.number_of_faults):
                    name = SET_PARAMS.Fault_names_values[index+1]
                    if multi_class:
                        Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, constellation = constellation, multi_class = True, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal)
                    else:
                        Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, constellation = constellation, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal)
                    X_list.append(X)    
                    Y_list.append(Y)

        else:
            for index in range(startNum, SET_PARAMS.number_of_faults):
                name = SET_PARAMS.Fault_names_values[index+1]
                if multi_class:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal)
                else:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal)
                X_list.append(X)    
                Y_list.append(Y)

        X = np.concatenate(X_list)
        Y = np.concatenate(Y_list)

    # step = []

    # initVal = Y[0]

    # for val in Y:
    #     if val != initVal:
    #         step.append(1)
    #     else:
    #         step.append(0)
    #     initVal = val

    # step = np.array(step)

    # step = step.reshape(step.shape[0],)

    Y = Y.reshape(Y.shape[0],)

    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    for depth in treeDepth:
        model = RandomForestClassifier(n_estimators = 100, max_depth = depth)

        model.fit(X_train, y_train)

        print(model.score(X_test, y_test))

        y_predicted = model.predict(X_test)

        cm = confusion_matrix(y_test, y_predicted)

        print('Random Forest', cm)

        path_to_folder = Path(path)
        path_to_folder.mkdir(exist_ok=True)

        if multi_class:
            pickle.dump(model, open(path + '/RandomForestMultiClass' + str(depth) + '.sav', 'wb'))
        else:
            pickle.dump(model, open(path + '/RandomForestBinaryClass' + str(depth) + '.sav', 'wb'))