import pickle
from Fault_prediction.Fault_utils import Dataset_order
from sklearn import tree
from sklearn.metrics import confusion_matrix
import numpy as np
from Simulation.Parameters import SET_PARAMS
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import dot2tex
import pydotplus

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def DecisionTreeAllAnomalies(path, featureExtractionMethod, constellation, multi_class, lowPredictionAccuracy, MovingAverage, includeAngularMomentumSensors, includeModelled, X, Y, NBType, treeDepth, ColumnNames, ClassNames, anomalyNames):
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

    # Beform a decision tree on the X and Y matrices
    # This must however include the moving average
    ColumnNames = ColumnNames.to_list()
    columnList = []

    for col in ColumnNames:
        if 'Moving Average' in col:
            columnList.append('DMD' + col.split('Moving Average')[1])
        elif 'Magnetometer' in col:
            columnList.append('Mag' + col.split('Magnetometer')[1])
        else:
            columnList.append(col)
        
    ColumnNames = columnList
    ClassNames = ["Normal", "Anomaly"]
    for depth in treeDepth:
        if lowPredictionAccuracy:
            clf = tree.DecisionTreeClassifier(max_depth = depth)
        else:
            clf = tree.DecisionTreeClassifier(max_depth = depth)

        # Split data into training and testing data
        mask = np.random.rand(len(X)) <= 0.6
        training_data = X[mask]
        testing_data = X[~mask]

        training_Y = Y[mask]
        testing_Y = Y[~mask]

        clf = clf.fit(training_data,training_Y)

        predict_y = clf.predict(testing_data)

        cm = confusion_matrix(testing_Y, predict_y)

        print('Decision Trees', cm)

        fontsize = 8

        path_to_folder = Path(path)
        path_to_folder.mkdir(exist_ok=True)

        figSize = (9, 6)
        file = str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/" + 'DecisionTree'

        # dot_data = tree.export_graphviz(clf, out_file=None, 
        #                         class_names = ClassNames, feature_names = ColumnNames, filled=True, max_depth = 2, label = 'all', rounded = True, proportion = True, 
        #                         special_characters=True) 
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_png(file + '.png')
        # graph.write_svg(file + ".svg")
        # graph.write_ps(file + ".ps")

        if lowPredictionAccuracy:
            pickle.dump(clf, open(path + '/DecisionTreesBinaryClassLowAccuracy' + str(depth) + '.sav', 'wb'))
            if SET_PARAMS.Visualize:
                fig = plt.figure(figsize = figSize, dpi = 300)
                # tree.plot_tree(clf, feature_names = ColumnNames, filled=True, max_depth = 2, fontsize = fontsize, label = 'all', rounded = True, proportion = True)
                tree.export_graphviz(clf, out_file = file, feature_names = ColumnNames, filled=True, max_depth = 2, label = 'all', rounded = True, proportion = True)
                dot2tex.dot2tex(file, format='tikz', crop=True)
                fig.tight_layout()
                # fig.savefig(Path(str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/" + 'DecisionTree.pgf'))
        elif multi_class and constellation:
            #! ClassNames = ["First", "Second", "Third", "Fourth", "Fifth"]
            pickle.dump(clf, open(path + '/ConstellationDecisionTreesMultiClass' + str(depth) + '.sav', 'wb'))
            if SET_PARAMS.Visualize:
                fig = plt.figure(figsize = figSize, dpi = 300)
                # tree.plot_tree(clf, feature_names = anomalyNames, filled=True, max_depth = 2, fontsize = fontsize, label = 'all', rounded = True, proportion = True)
                tree.export_graphviz(clf, out_file = file, feature_names = ColumnNames, filled=True, max_depth = 2, label = 'all', rounded = True, proportion = True)
                dot2tex.dot2tex(file, format='tikz', crop=True)
                fig.tight_layout()
                # fig.savefig(Path(str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/" + 'DecisionTree.pgf'))
        elif constellation:
            pickle.dump(clf, open(path + '/ConstellationDecisionTreesBinaryClass' + str(depth) + '.sav', 'wb'))

            if SET_PARAMS.Visualize:
                fig = plt.figure(figsize = figSize, dpi = 300)
                # tree.plot_tree(clf, class_names = ClassNames, feature_names = ColumnNames, filled=True, max_depth = 2, fontsize = fontsize,  precision = 2, label = 'all', rounded = True, proportion = True)
                tree.export_graphviz(clf, out_file = file, feature_names = ColumnNames, filled=True, max_depth = 2, label = 'all', rounded = True, proportion = True)
                dot2tex.dot2tex(file, format='tikz', crop=True)
                fig.tight_layout()
                # fig.savefig(Path(str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/" + 'DecisionTree.pgf'))

        elif multi_class:
            pickle.dump(clf, open(path + '/DecisionTreesMultiClass' + str(depth) + '.sav', 'wb'))
            if SET_PARAMS.Visualize:
                fig = plt.figure(figsize = figSize, dpi = 300)
                # tree.plot_tree(clf, class_names = anomalyNames, feature_names = ColumnNames, filled=True, max_depth = 2, fontsize = fontsize,  precision = 2, label = 'all', rounded = True, proportion = True)
                tree.export_graphviz(clf, out_file = file, feature_names = ColumnNames, filled=True, max_depth = 2, label = 'all', rounded = True, proportion = True)
                dot2tex.dot2tex(file, format='tikz', crop=True)
                fig.tight_layout()
                # fig.savefig(Path(str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/" + 'DecisionTree.pgf'))
        else:
            pickle.dump(clf, open(path + '/DecisionTreesBinaryClass' + str(depth) + '.sav', 'wb'))

        #     if SET_PARAMS.Visualize:
        #         fig = plt.figure(figsize = figSize, dpi = 300)
        #         # tree.plot_tree(clf, class_names = ClassNames, feature_names = ColumnNames, filled=True, max_depth = 2, fontsize = fontsize,  precision = 2, label = 'all', rounded = True, proportion = True)
        #         tree.export_graphviz(clf, out_file = file, feature_names = ColumnNames, filled=True, max_depth = 2, label = 'all', rounded = True, proportion = True)
        #         dot2tex.dot2tex(file, format='tikz', crop=True)
        #         fig.tight_layout()
        #         # fig.savefig(Path(str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/" + 'DecisionTree.pgf'))
