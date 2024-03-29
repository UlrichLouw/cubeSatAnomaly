import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from Simulation.Parameters import SET_PARAMS
from Fault_prediction.Fault_utils import Dataset_order
from sklearn.metrics import confusion_matrix
import pickle
from pathlib import Path

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

np.random.seed(42)


def LOF(path, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames):
    
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

    # fit the model for outlier detection (default)
    clf = LocalOutlierFactor(n_neighbors=1000, contamination=0.01, novelty = True, n_jobs = -1)

    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).
    clf.fit(training_data)  
    
    y_pred = clf.predict(testing_data)

    y_pred = (y_pred - 1)*(-1/2)

    cm = confusion_matrix(testing_Y, y_pred)

    print('LOF', cm)

    n_errors = (y_pred != testing_Y).sum()
    X_scores = clf.negative_outlier_factor_

    print(X_scores)

    # plt.title("Local Outlier Factor (LOF)")
    # plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
    # # plot circles with radius proportional to the outlier scores
    # radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    # plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
    #             facecolors='none', label='Outlier scores')
    # plt.axis('tight')
    # plt.xlim((-5, 5))
    # plt.ylim((-5, 5))
    # plt.xlabel("prediction errors: %d" % (n_errors))
    # legend = plt.legend(loc='upper left')
    # legend.legendHandles[0]._sizes = [10]
    # legend.legendHandles[1]._sizes = [20]

    # plt.savefig(Path(str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/" + 'LOF.pgf'))

    # plt.close()

    if multi_class:
        pickle.dump(clf, open(path + '/LOFMultiClass.sav', 'wb'))
    else:
        pickle.dump(clf, open(path + '/LOFBinaryClass.sav', 'wb'))

