import numpy as np
from sqlalchemy import true
from Simulation.Parameters import SET_PARAMS

from Fault_prediction.Feature_extraction import DMD
from Fault_prediction.Supervised_Learning import DecisionForests
from Fault_prediction.Supervised_Learning import Random_Forest
from Fault_prediction.Unsupervised_Learning import Isolation_Forest
from Fault_prediction.Unsupervised_Learning import KMeansCluster
from Fault_prediction.Supervised_Learning import SupportVectorMachines
from Fault_prediction.Supervised_Learning import NaiveBayes
from Fault_prediction.Unsupervised_Learning import LOF
#from Fault_prediction.Unsupervised_Learning import Extended_Isolation_Forest
import sys
import pickle
import matplotlib.pyplot as plt
from Fault_prediction.Fault_utils import Dataset_order
from sklearn import tree
import multiprocessing

np.set_printoptions(threshold=500)

if __name__ == '__main__':
    SET_PARAMS.Visualize = False
    SET_PARAMS.sensor_number = 0
    SET_PARAMS.Kalman_filter_use = "EKF"
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.FeatureExtraction = "None"
    SET_PARAMS.SensorPredictor = "None"
    SET_PARAMS.SensorRecoveror = "None" 
    SET_PARAMS.SensorIsolator = "None"
    SET_PARAMS.number_of_faults = 3
    SET_PARAMS.Number_of_satellites = 100
    SET_PARAMS.Model_or_Measured = "ORC"
    singleAnomaly = False
    constellation = False
    multi_class = True
    lowPredictionAccuracy = False
    MovingAverage = False
    includeAngularMomentumSensors = False
    includeModelled = False

    if SET_PARAMS.FeatureExtraction == "DMD":
        MovingAverage = True

    featureExtractionMethod = SET_PARAMS.FeatureExtraction
    
    treeDepth = [100] #5, 10, 20, 

    GenericPath = "FeatureExtraction-" + SET_PARAMS.FeatureExtraction + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+ SET_PARAMS.Model_or_Measured +"/" +"General CubeSat Model/"
    
    if constellation:
        GenericPath = "Constellation/" + "FeatureExtraction-" + SET_PARAMS.FeatureExtraction + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+ SET_PARAMS.Model_or_Measured +"/" +"General CubeSat Model/"
    
    SET_PARAMS.path = SET_PARAMS.path + GenericPath
    SET_PARAMS.numberOfSensors = 3
    # Compute the A and B matrix to estimate X
    # for i in range(SET_PARAMS.numberOfSensors):
    #     print(i)
    #     DMD.MatrixAB(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', includeModelled = includeModelled)
    #     SET_PARAMS.sensor_number += 1
    # SET_PARAMS.sensor_number = "ALL"
    # DMD.MatrixAB(path = SET_PARAMS.pathHyperParameters + 'PhysicsEnabledDMDMethod', includeModelled = includeModelled)
    # DecisionTree training
    NBType = ["Bernoulli", "Gaussian"] #"Gaussian", 

    tag = SET_PARAMS.FeatureExtraction

    X_list = []
    Y_list = []

    pathFiles = SET_PARAMS.path

    buffer = False

    SET_PARAMS.buffer_size = 100

    if multi_class:
        ignoreNormal = False
        startNum = 0
    else:
        ignoreNormal = False
        startNum = 0

    # if singleAnomaly:
    #     startNum = 1
    #     SET_PARAMS.number_of_faults = startNum + 1
    #     tag = tag + "_" + SET_PARAMS.Fault_names_values[startNum+1]

    anomalyNames = []

    if constellation:
        for satNum in range(SET_PARAMS.Number_of_satellites):
            print(satNum)
            SET_PARAMS.path = pathFiles + str(satNum) + "/"
            for index in range(startNum, SET_PARAMS.number_of_faults):
                name = SET_PARAMS.Fault_names_values[index+1]
                if multi_class:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, constellation = constellation, multi_class = True, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal, featureExtractionMethod = SET_PARAMS.FeatureExtraction)
                else:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, constellation = constellation, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal, featureExtractionMethod = SET_PARAMS.FeatureExtraction)
                X_list.append(X)    
                Y_list.append(Y)

    else:
        for index in range(startNum, SET_PARAMS.number_of_faults):
            name = SET_PARAMS.Fault_names_values[index+1]
            print(name)
            anomalyNames.append(name)
            if multi_class:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal, featureExtractionMethod = SET_PARAMS.FeatureExtraction)
            else:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, ignoreNormal = ignoreNormal, featureExtractionMethod = SET_PARAMS.FeatureExtraction)
            X_list.append(X)    
            Y_list.append(Y)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)        

    threads = []

    # t = multiprocessing.Process(target=LOF.LOF, args=(SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames))
    # threads.append(t)
    # t.start()

    # t = multiprocessing.Process(target=KMeansCluster.KMeanBinary, args=(SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames))
    # threads.append(t)
    # t.start()

    # t = multiprocessing.Process(target=NaiveBayes.NB, args=(SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames))
    # threads.append(t)
    # t.start()

    # t = multiprocessing.Process(target=Isolation_Forest.IsoForest, args=(SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames))
    # threads.append(t)
    # t.start()

    t = multiprocessing.Process(target=SupportVectorMachines.SVM, args=(SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames))
    threads.append(t)
    t.start()

    t = multiprocessing.Process(target=DecisionForests.DecisionTreeAllAnomalies, args=(SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames, anomalyNames))
    threads.append(t)
    t.start()

    t = multiprocessing.Process(target=Random_Forest.Random_Forest, args=(SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames))
    threads.append(t)
    t.start()

    for process in threads:     
        process.join()

    threads.clear()

    # print('Local Outlier Factor')
    # LOF.LOF(path = SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod = featureExtractionMethod, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, X = X, Y = Y)
    # print('Naive Bayes')
    # NaiveBayes.NB(path = SET_PARAMS.pathHyperParameters + tag, NBType = nbtype, featureExtractionMethod = featureExtractionMethod, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, X = X, Y = Y)
    # print('Support Vector Machines')
    # SupportVectorMachines.SVM(path = SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod = featureExtractionMethod, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, X = X, Y = Y)
    # print('Isolation Forests')
    # Isolation_Forest.IsoForest(path = SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod = featureExtractionMethod, treeDepth = treeDepth, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, X = X, Y = Y)
    # # Extended_Isolation_Forest.IsoForest(path = SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod = featureExtractionMethod, treeDepth = treeDepth, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled)
    # print('K Means')
    # KMeansCluster.KMeanBinary(path = SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod = featureExtractionMethod, treeDepth = treeDepth, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, X = X, Y = Y)
    # print('Decision Trees')
    # DecisionForests.DecisionTreeAllAnomalies(path = SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod = featureExtractionMethod, treeDepth = treeDepth, constellation = constellation, multi_class = multi_class, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, X = X, Y = Y)
    # print('Random Forests')
    # Random_Forest.Random_Forest(path = SET_PARAMS.pathHyperParameters + tag, featureExtractionMethod = featureExtractionMethod, treeDepth = treeDepth, constellation = constellation, multi_class = False, lowPredictionAccuracy = lowPredictionAccuracy, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomentumSensors, includeModelled = includeModelled, X = X, Y = Y)