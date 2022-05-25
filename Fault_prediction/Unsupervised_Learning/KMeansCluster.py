from sklearn.cluster import KMeans
import numpy as np
from Simulation.Parameters import SET_PARAMS
from Fault_prediction.Fault_utils import Dataset_order
from sklearn.metrics import confusion_matrix
import pickle

def KMeanBinary(path, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames):
    X_list = []
    Y_list = []

    pathFiles = SET_PARAMS.path

    buffer = True
    SET_PARAMS.buffer_size = 2

    if multi_class:
        ignoreNormal = True
        startNum = 1
    else:
        ignoreNormal = False
        startNum = 0

    # if (X == None).any() or (Y == None).any():
    #     if constellation:
    #         for satNum in range(SET_PARAMS.Number_of_satellites):
    #             print(satNum)
    #             SET_PARAMS.path = pathFiles + str(satNum) + "/"
    #             for index in range(startNum, SET_PARAMS.number_of_faults):
    #                 name = SET_PARAMS.Fault_names_values[index+1]
    #                 if multi_class:
    #                     Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, constellation = constellation, multi_class = True, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal)
    #                 else:
    #                     Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, constellation = constellation, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal)
    #                 X_list.append(X)    
    #                 Y_list.append(Y)

    #     else:
    #         for index in range(startNum, SET_PARAMS.number_of_faults):
    #             name = SET_PARAMS.Fault_names_values[index+1]
    #             if multi_class:
    #                 Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal)
    #             else:
    #                 Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal)
    #             X_list.append(X)    
    #             Y_list.append(Y)

    #     X = np.concatenate(X_list)
    #     Y = np.concatenate(Y_list)


    # Split data into training and testing data
    mask = np.random.rand(len(X)) <= 0.6
    training_data = X[mask]
    testing_data = X[~mask]

    training_Y = Y[mask]
    testing_Y = Y[~mask]

    kmeans = KMeans(n_clusters = 2, random_state = 0, n_init = 50, max_iter = 500).fit(X)

    yPred = kmeans.predict(testing_data)

    cm = confusion_matrix(testing_Y, yPred)

    print('KMeans', cm)


    if multi_class:
        pickle.dump(kmeans, open(path + '/KMeansClustering' + featureExtractionMethod  + 'MultiClass.sav', 'wb'))
    else:
        pickle.dump(kmeans, open(path + '/KMeansClustering' + featureExtractionMethod  + 'BinaryClass.sav', 'wb'))