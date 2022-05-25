from re import S
import matplotlib
from Simulation.Parameters import SET_PARAMS
import pandas as pd
from pathlib import Path
import numpy as np
from Extra.util import createParams

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def GetData(path, index, n, all = False, first = False):
    Dataframe = pd.read_csv(path, low_memory=False)

    if all:
        Datapgf = Dataframe
    elif first:
        Datapgf = Dataframe[:int((n)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))]
    else:
        Datapgf = Dataframe[int((SET_PARAMS.Number_of_orbits-n)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)):]

    return Datapgf


def dataMethodPlot(recovery, extraction, isolation, prediction, bufferValue, bufferStep, recoveryBuffer, predictionBuffer, perfectNoFailurePrediction, legend, index, recoverMethodsWithoutPrediction, Data):
    method = extraction + str(prediction) + str(isolation) + recovery + SET_PARAMS.Fault_names_values[index] 
    if predictionBuffer:
        method += str(bufferValue) + str(bufferStep) + recoveryBuffer

    if perfectNoFailurePrediction:
        method += str("perfectNoFailurePrediction")

    plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

    if legend == "recovery":
        legendName = str(recovery)
    elif legend == "prediction":
        legendName = str(prediction)
    elif legend == "bufferValue":
        legendName = str(bufferValue)
    elif legend == "isolation":
        legendName = str(isolation)
    elif legend == "bufferStep":
        legendName = str(bufferStep)
    elif legend == "extraction":
        legendName = str(extraction)
    elif legend == "predictionBuffer":
        legendName = str(predictionBuffer)
    elif legend == "perfectNoFailurePrediction":
        legendName = str(perfectNoFailurePrediction)
    
    return legendName, plotData

def SummaryPlots(legend, BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction, includeNone, bbox_to_anchor, loc, plotColumns, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index, Number, Number_of_orbits = 30, ALL = True, first = False, width = 8.0, height = 6.0, groupBy = "Recovery", uniqueTag = ""):
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.Model_or_Measured = "ORC"
    SET_PARAMS.Number_of_orbits = Number_of_orbits
    SET_PARAMS.save_as = ".csv"
    SET_PARAMS.Low_Aerodynamic_Disturbance = False

    if index == 1:
        predictionMethods = ["None"]
        isolationMethods = ["None"]
        recoveryMethods = ["None"]
        recoverMethodsWithoutPrediction = ["None"]

    cm = 1/2.54

    path_of_execution = str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/TexFigures"

    Path(path_of_execution).mkdir(parents = True, exist_ok=True)

    plt = matplotlib.pyplot

    for name in plotColumns:
        legendLines = []
        legendNames = []

        path = "Data files/Summary/" + name + "/" + SET_PARAMS.Fault_names_values[index] + ".csv"

        Datapgf = GetData(path, index, n = Number, all = ALL, first = first) 

        Data = Datapgf[Datapgf.columns[Datapgf.iloc[1] == "Mean"]].copy()

        Data["Unnamed: 0"] = Datapgf["Unnamed: 0"].copy()
        
        if groupBy == "recovery":
            for recovery in recoveryMethods:

                plt.figure(figsize = (width*cm, height*cm))

                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                pathParams = createParams(featureExtractionMethods, isolationMethods, predictionMethods, BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction)
                for extraction, isolation, prediction, bufferValue, bufferStep, recoveryBuffer, predictionBuffer, perfectNoFailurePrediction in pathParams:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None" and not predictionBuffer and not perfectNoFailurePrediction) or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                        legendName, plotData = dataMethodPlot(recovery = recovery, extraction = extraction, isolation = isolation,
                                                            prediction = prediction, bufferValue = bufferValue, bufferStep = bufferStep, 
                                                            recoveryBuffer = recoveryBuffer, predictionBuffer = predictionBuffer, 
                                                            perfectNoFailurePrediction = perfectNoFailurePrediction, legend = legend, 
                                                            index = index, recoverMethodsWithoutPrediction = recoverMethodsWithoutPrediction, Data = Data)
                        legendNames.append(legendName)  
                        plt.plot(range(len(plotData)), plotData)

                    if includeNone:
                        method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                        plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                        plt.plot(range(len(plotData)), plotData)
                        legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(recovery)

                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + name + uniqueTag+ '.pgf'))

                plt.close()

        elif groupBy == "prediction":
            for prediction in predictionMethods:

                plt.figure(figsize = (width*cm, height*cm))

                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                pathParams = createParams(featureExtractionMethods, isolationMethods, recoveryMethods, BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction)
                for extraction, isolation, recovery, bufferValue, bufferStep, recoveryBuffer, predictionBuffer, perfectNoFailurePrediction in pathParams:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None" and not predictionBuffer and not perfectNoFailurePrediction) or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                        legendName, plotData = dataMethodPlot(recovery = recovery, extraction = extraction, isolation = isolation,
                                                            prediction = prediction, bufferValue = bufferValue, bufferStep = bufferStep, 
                                                            recoveryBuffer = recoveryBuffer, predictionBuffer = predictionBuffer, 
                                                            perfectNoFailurePrediction = perfectNoFailurePrediction, legend = legend, 
                                                            index = index, recoverMethodsWithoutPrediction = recoverMethodsWithoutPrediction, Data = Data)
                        legendNames.append(legendName)  
                        plt.plot(range(len(plotData)), plotData)

                    if includeNone:
                        method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                        plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                        plt.plot(range(len(plotData)), plotData)
                        legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(prediction)

                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + name + uniqueTag+ '.pgf'))

                plt.close()

        elif groupBy == "bufferValue":
            for bufferValue in BufferValue:

                plt.figure(figsize = (width*cm, height*cm))

                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                pathParams = createParams(featureExtractionMethods, isolationMethods, predictionMethods, recoveryMethods, BufferStep, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction)
                for extraction, isolation, prediction, recovery, bufferStep, recoveryBuffer, predictionBuffer, perfectNoFailurePrediction in pathParams:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None" and not predictionBuffer and not perfectNoFailurePrediction) or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                        legendName, plotData = dataMethodPlot(recovery = recovery, extraction = extraction, isolation = isolation,
                                                            prediction = prediction, bufferValue = bufferValue, bufferStep = bufferStep, 
                                                            recoveryBuffer = recoveryBuffer, predictionBuffer = predictionBuffer, 
                                                            perfectNoFailurePrediction = perfectNoFailurePrediction, legend = legend, 
                                                            index = index, recoverMethodsWithoutPrediction = recoverMethodsWithoutPrediction, Data = Data)
                        legendNames.append(legendName)  
                        plt.plot(range(len(plotData)), plotData)

                    if includeNone:
                        method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                        plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                        plt.plot(range(len(plotData)), plotData)
                        legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(bufferValue)

                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + name + uniqueTag+ '.pgf'))

                plt.close()

        elif groupBy == "isolation":
            for isolation in isolationMethods:
                plt.figure(figsize = (width*cm, height*cm))

                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                pathParams = createParams(featureExtractionMethods, recoveryMethods, predictionMethods, BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction)
                for extraction, recovery, prediction, bufferValue, bufferStep, recoveryBuffer, predictionBuffer, perfectNoFailurePrediction in pathParams:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None" and not predictionBuffer and not perfectNoFailurePrediction) or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                        legendName, plotData = dataMethodPlot(recovery = recovery, extraction = extraction, isolation = isolation,
                                                            prediction = prediction, bufferValue = bufferValue, bufferStep = bufferStep, 
                                                            recoveryBuffer = recoveryBuffer, predictionBuffer = predictionBuffer, 
                                                            perfectNoFailurePrediction = perfectNoFailurePrediction, legend = legend, 
                                                            index = index, recoverMethodsWithoutPrediction = recoverMethodsWithoutPrediction, Data = Data)
                        legendNames.append(legendName)  
                        plt.plot(range(len(plotData)), plotData)

                    if includeNone:
                        method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                        plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                        plt.plot(range(len(plotData)), plotData)
                        legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(isolation)

                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + name + uniqueTag+ '.pgf'))

                plt.close()

        elif groupBy == "bufferStep":
            for bufferStep in BufferStep:
                plt.figure(figsize = (width*cm, height*cm))

                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                pathParams = createParams(featureExtractionMethods, isolationMethods, predictionMethods, BufferValue, recoveryMethods, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction)
                for extraction, isolation, prediction, bufferValue, recovery, recoveryBuffer, predictionBuffer, perfectNoFailurePrediction in pathParams:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None" and not predictionBuffer and not perfectNoFailurePrediction) or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                        legendName, plotData = dataMethodPlot(recovery = recovery, extraction = extraction, isolation = isolation,
                                                            prediction = prediction, bufferValue = bufferValue, bufferStep = bufferStep, 
                                                            recoveryBuffer = recoveryBuffer, predictionBuffer = predictionBuffer, 
                                                            perfectNoFailurePrediction = perfectNoFailurePrediction, legend = legend, 
                                                            index = index, recoverMethodsWithoutPrediction = recoverMethodsWithoutPrediction, Data = Data)
                        legendNames.append(legendName)  
                        plt.plot(range(len(plotData)), plotData)

                    if includeNone:
                        method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                        plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                        plt.plot(range(len(plotData)), plotData)
                        legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(bufferStep)

                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + name + uniqueTag+ '.pgf'))

                plt.close()

        elif groupBy == "extraction":
            for extraction in featureExtractionMethods:

                plt.figure(figsize = (width*cm, height*cm))

                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                pathParams = createParams(recoveryMethods, isolationMethods, predictionMethods, BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction)
                for recovery, isolation, prediction, bufferValue, bufferStep, recoveryBuffer, predictionBuffer, perfectNoFailurePrediction in pathParams:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None" and not predictionBuffer and not perfectNoFailurePrediction) or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                        legendName, plotData = dataMethodPlot(recovery = recovery, extraction = extraction, isolation = isolation,
                                                            prediction = prediction, bufferValue = bufferValue, bufferStep = bufferStep, 
                                                            recoveryBuffer = recoveryBuffer, predictionBuffer = predictionBuffer, 
                                                            perfectNoFailurePrediction = perfectNoFailurePrediction, legend = legend, 
                                                            index = index, recoverMethodsWithoutPrediction = recoverMethodsWithoutPrediction, Data = Data)
                        legendNames.append(legendName)  
                        plt.plot(range(len(plotData)), plotData)

                    if includeNone:
                        method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                        plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                        plt.plot(range(len(plotData)), plotData)
                        legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(extraction)

                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + name + uniqueTag+ '.pgf'))

                plt.close()

        elif groupBy == "predictionBuffer":
            for predictionBuffer in PredictionBuffer:

                plt.figure(figsize = (width*cm, height*cm))

                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                pathParams = createParams(featureExtractionMethods, isolationMethods, predictionMethods, BufferValue, BufferStep, RecoveryBuffer, recoveryMethods, PerfectNoFailurePrediction)
                for extraction, isolation, prediction, bufferValue, bufferStep, recoveryBuffer, recovery, perfectNoFailurePrediction in pathParams:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None" and not predictionBuffer and not perfectNoFailurePrediction) or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                        legendName, plotData = dataMethodPlot(recovery = recovery, extraction = extraction, isolation = isolation,
                                                            prediction = prediction, bufferValue = bufferValue, bufferStep = bufferStep, 
                                                            recoveryBuffer = recoveryBuffer, predictionBuffer = predictionBuffer, 
                                                            perfectNoFailurePrediction = perfectNoFailurePrediction, legend = legend, 
                                                            index = index, recoverMethodsWithoutPrediction = recoverMethodsWithoutPrediction, Data = Data)
                        legendNames.append(legendName)  
                        plt.plot(range(len(plotData)), plotData)

                    if includeNone:
                        method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                        plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                        plt.plot(range(len(plotData)), plotData)
                        legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(predictionBuffer)

                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + name + uniqueTag+ '.pgf'))

                plt.close()

        elif groupBy == "perfectNoFailurePrediction":
            for perfectNoFailurePrediction in PerfectNoFailurePrediction:

                plt.figure(figsize = (width*cm, height*cm))

                plt.grid(visible = True, which = 'both')

                plt.xlabel("Number of Orbits", fontsize = int(width))

                if "Metric" in name:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                pathParams = createParams(featureExtractionMethods, isolationMethods, predictionMethods, BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, recoveryMethods)
                for extraction, isolation, prediction, bufferValue, bufferStep, recoveryBuffer, predictionBuffer, recovery in pathParams:
                    if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None" and not predictionBuffer and not perfectNoFailurePrediction) or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                        legendName, plotData = dataMethodPlot(recovery = recovery, extraction = extraction, isolation = isolation,
                                                            prediction = prediction, bufferValue = bufferValue, bufferStep = bufferStep, 
                                                            recoveryBuffer = recoveryBuffer, predictionBuffer = predictionBuffer, 
                                                            perfectNoFailurePrediction = perfectNoFailurePrediction, legend = legend, 
                                                            index = index, recoverMethodsWithoutPrediction = recoverMethodsWithoutPrediction, Data = Data)
                        legendNames.append(legendName)  
                        plt.plot(range(len(plotData)), plotData)

                    if includeNone:
                        method = "DMDNoneNoneNone" + SET_PARAMS.Fault_names_values[index]

                        plotData = [float(x) for x in Data[Data["Unnamed: 0"] == method].values[0] if x != method]

                        plt.plot(range(len(plotData)), plotData)
                        legendNames.append(str("None"))

                if len(legendNames) > 1:
                    plt.legend(legendNames, loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Summary/" + str(perfectNoFailurePrediction)

                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + name + uniqueTag+ '.pgf'))

                plt.close()