import LatexPlots.Metrics as Metrics
import LatexPlots.Vectors as Vectors
import LatexPlots.Summary as Summary


if __name__ == '__main__':
    featureExtractionMethods = ["DMD"]
    isolationMethods = ["None", "OnlySun"]
    recoveryMethods = ["EKF-ignore"] #, "EKF-combination", "EKF-reset", "EKF-top2"] #,  "EKF-top2"
    recoverMethodsWithoutPrediction = ["EKF-top2"]
    plotColumns = ["Prediction Accuracy", "Estimation Metric", "Pointing Metric"] #"Prediction Accuracy", 
    treeDepth = [20] # [5, 10, 20, 100]

    predictionMethod = ["RandomForest"] #"DecisionTrees", "RandomForest"]  #, "SVM", "NaiveBayesBernoulli", "NaiveBayesGaussian", "Isolation_Forest", "PERFECT", 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5, 99.0, 99.5, 99.9]

    RecoveryBuffer = ["EKF-top2"]
    PredictionBuffer = [False]
    BufferValue = [10]
    BufferStep = [0.9]

    perfectNoFailurePrediction = [False]

    groupBy = "recovery"
    tag = "RF20DT20"
    includeNone = False
    bbox_to_anchor = (0.5, 0.5, 0.5, 0.5)
    legend = "prediction"

    loc = 1

    grid = False
    plotStyle = "Line"
    linewidth = 1

    predictionMethods = []

    for prediction in predictionMethod:
        if prediction == "DecisionTrees" or prediction == "RandomForest":
            for depth in treeDepth:
                predictionMethods.append(prediction + str(depth))
        else:
            predictionMethods.append(prediction)

    featureExtractionMethods = ["DMD"]
    isolationMethods = ["None"]
    recoveryMethods = ["None"] #, "EKF-combination", "EKF-reset", "EKF-top2"] #,  "EKF-top2"
    recoverMethodsWithoutPrediction = ["None"]
    plotColumns = ["Prediction Accuracy", "Estimation Metric", "Pointing Metric"]
    predictionMethods = ["None"]

    index = 1

    plotColumns = ["SolarPanelDipole Torques", "Magnetometer"]

    # plotColumns = ["Earth"] #, "Magnetometer"]

    # plotColumns = ["Sun", "Magnetometer"]

    plotColumns = ["Earth"]

    # Metrics.MetricPlots(BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, perfectNoFailurePrediction, bbox_to_anchor, loc, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index = index, Number = 2, Number_of_orbits = 30, first = True, ALL = False, width = 8.0, height = 6.0, linewidth = linewidth, grid = grid, plotStyle = plotStyle)
    Vectors.VectorPlots(BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, perfectNoFailurePrediction, bbox_to_anchor, loc, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index = index, Number = 2, Number_of_orbits = 5, first = True, ALL = False, width = 8.0, height = 6.0, plotColumns = plotColumns)
    # Summary.SummaryPlots(legend, BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, perfectNoFailurePrediction, includeNone, bbox_to_anchor, loc, plotColumns, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index = index, Number = 30, Number_of_orbits = 30, first = True, ALL = True, width = 8.0, height = 6.0, groupBy = groupBy, uniqueTag = tag)



    # faultNames = ["None",
    #             "Reflection",
    #             "solarPanelDipole",
    #             'catastrophicReactionWheel']