import LatexPlots.Metrics as Metrics
import LatexPlots.Vectors as Vectors
import LatexPlots.Summary as Summary


if __name__ == '__main__':
    featureExtractionMethods = ["None"]
    predictionMethod = ["None"] #"PERFECT", 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5, 99.0, 99.5, 99.9]
    isolationMethod =  ["None"] #, "SVM"]#[] #, "RandomForest", "SVM"]
    recoveryMethods = ['None'] #, 'EKF-combination', 'EKF-reset'] #, "EKF-top2"] #,  "EKF-top2"
    recoverMethodsWithoutPrediction = ["None"] #, "EKF-top2"]
    plotColumns = ["Isolation Accuracy", "Prediction Accuracy", "Estimation Metric", "Pointing Metric"] #"Prediction Accuracy", 
    treeDepth = [100] # [5, 10, 20, 100]


    RecoveryBuffer = ["EKF-top2"]
    PredictionBuffer = [False]
    BufferValue = [10]
    BufferStep = [0.9]

    perfectNoFailurePrediction = [False]

    groupBy = "isolation"
    tag = "DifferentPredictionMethods"
    includeNone = False
    bbox_to_anchor = (0.5, 0.5, 0.5, 0.5)
    legend = "prediction"

    loc = 1

    grid = True
    plotStyle = "Line"
    linewidth = 1

    predictionMethods = []
    isolationMethods = []

    for prediction in predictionMethod:
        if prediction == "DecisionTrees" or prediction == "RandomForest":
            for depth in treeDepth:
                predictionMethods.append(prediction + str(depth))
        else:
            predictionMethods.append(prediction)

    for isolation in isolationMethod:
        if isolation == "DecisionTrees" or isolation == "RandomForest":
            for depth in treeDepth:
                isolationMethods.append(isolation + str(depth))
        else:
            isolationMethods.append(isolation)



    plotColumns = ["Prediction Accuracy", "Estimation Metric", "Pointing Metric"]

    index = 3

    # plotColumns = ["SolarPanelDipole Torques", "Magnetometer"]

    # plotColumns = ["Earth"] #, "Magnetometer"]

    # plotColumns = ["Sun", "Magnetometer", "Earth", "Angular momentum of wheels", "Aerodynamic Torques", "Wheel disturbance Torques", "Gravity Gradient Torques", "Magnetic Control Torques", "Wheel Control Torques", "SolarPanelDipole Torques"]

    # plotColumns = ["DMD"] #, "DMD"]

    width = 12
    height = 9

    singleValue = ["DMD"] #, "DMD"]

    Metrics.MetricPlots(BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, perfectNoFailurePrediction, bbox_to_anchor, loc, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index = index, Number = 2, Number_of_orbits = 30, first = True, ALL = False, width = width, height = height, linewidth = linewidth, grid = grid, plotStyle = plotStyle)
    # Vectors.VectorPlots(BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, perfectNoFailurePrediction, bbox_to_anchor, loc, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index = index, Number = 2, Number_of_orbits = 5, first = True, ALL = False, width = width, height = height, plotColumns = plotColumns, singleValue = singleValue)
    # Summary.SummaryPlots(legend, BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, perfectNoFailurePrediction, includeNone, bbox_to_anchor, loc, plotColumns, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index = index, Number = 30, Number_of_orbits = 30, first = True, ALL = True, width = width, height = height, groupBy = groupBy, uniqueTag = tag)



    # faultNames = ["None",
    #             "Reflection",
    #             "solarPanelDipole",
    #             'catastrophicReactionWheel']