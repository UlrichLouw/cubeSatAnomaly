from Fault_prediction.Fault_utils import Dataset_order
from sklearn.metrics import confusion_matrix
import numpy as np
from Simulation.Parameters import SET_PARAMS
from sklearn import svm
import pickle
from pathlib import Path

def SVM(path, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames):
    X_list = []
    Y_list = []

    pathFiles = SET_PARAMS.path

    buffer = False

    SET_PARAMS.buffer_size = 100

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

    Y = Y.reshape(Y.shape[0],)

    # Split data into training and testing data
    mask = np.random.rand(len(X)) <= 0.6
    training_data = X[mask]
    testing_data = X[~mask]

    training_Y = Y[mask]
    testing_Y = Y[~mask]

    if multi_class:
        clf = svm.SVC(decision_function_shape='ovr')
    else:
        clf = svm.SVC()

    clf.fit(training_data,training_Y)

    y_pred = clf.predict(testing_data)

    cm = confusion_matrix(testing_Y, y_pred)

    print('SVM', cm)

    path_to_folder = Path(path)
    path_to_folder.mkdir(exist_ok=True)

    if multi_class:
        pickle.dump(clf, open(path + '/StateVectorMachineMultiClass.sav', 'wb'))
    else:
        pickle.dump(clf, open(path + '/StateVectorMachineBinaryClass.sav', 'wb'))