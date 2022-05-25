from glob import escape
import matplotlib
from Simulation.Parameters import SET_PARAMS
import pandas as pd
from pathlib import Path
from Extra.util import createParams
import os

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def GetData(path, index, n, all = False, first = False):
    try:
        Dataframe = pd.read_csv(Path(path), low_memory=False)

        if all:
            Datapgf = Dataframe
        elif first:
            Datapgf = Dataframe[:int((n)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))]
        else:
            Datapgf = Dataframe[int((SET_PARAMS.Number_of_orbits-n)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)):]

        Datapgf.reset_index(drop=True, inplace = True)

        return Datapgf
    except:
        return None

def MetricPlots(BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction, bbox_to_anchor, loc, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction, index, Number, Number_of_orbits = 30, ALL = True, first = False, width = 8.0, height = 6.0, linewidth = 1.5, grid = True, plotStyle = 'Line'):
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

    includeNone = True

    cm = 1/2.54

    plotColumns = ["Prediction Accuracy", "Estimation Metric", "Pointing Metric"]
    # plotColumns = ["Estimation Metric"]

    path_of_execution = str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/TexFigures"

    Path(path_of_execution).mkdir(parents = True, exist_ok=True)

    plt = matplotlib.pyplot

    satelliteFDIRParams = createParams(PredictionBuffer, BufferValue, BufferStep, PerfectNoFailurePrediction, RecoveryBuffer, featureExtractionMethods, recoveryMethods, predictionMethods, isolationMethods)


    for predictionBuffer, bufferValue, bufferStep, perfectNoFailurePrediction, recoveryBuffer, extraction, recovery, prediction, isolation in satelliteFDIRParams:
        if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
            SET_PARAMS.FeatureExtraction = extraction
            SET_PARAMS.SensorPredictor = str(prediction)
            SET_PARAMS.SensorIsolator = str(isolation)
            SET_PARAMS.SensorRecoveror = recovery
            GenericPath = "FeatureExtraction-" + str(SET_PARAMS.FeatureExtraction) + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/"+ "General CubeSat Model/"
            
            if SET_PARAMS.Low_Aerodynamic_Disturbance:
                GenericPath = "Low_Disturbance/" + GenericPath

            if predictionBuffer:
                GenericPath += "BufferValue-" + str(bufferValue) + "BufferStep-" + str(bufferStep) + str(recoveryBuffer) + "/"

            if perfectNoFailurePrediction:
                GenericPath = GenericPath + "PerfectNoFailurePrediction/"
            
            path = "Data files/"+ GenericPath + SET_PARAMS.Fault_names_values[index] + ".csv.gz"
            # path = Path(path)
            Datapgf = GetData(path, index, n = Number, all = ALL, first = first) 
            
            currenctSunStatus = Datapgf.loc[0, "Sun in view"]

            SunInView = []
            Eclipse = []
            
            for ind in Datapgf.index:
                if Datapgf.loc[ind, "Sun in view"] != currenctSunStatus:
                    if Datapgf.loc[ind, "Sun in view"]:
                        SunInView.append(ind)
                    else:
                        Eclipse.append(ind)

                currenctSunStatus = Datapgf.loc[ind, "Sun in view"]

            for col in plotColumns:
                plt.figure(figsize = (width*cm, height*cm))

                if plotStyle == "Scatter":
                    plt.scatter(range(len(Datapgf[[col]])), Datapgf[[col]], linewidth = linewidth)
                else:
                    plt.plot(range(len(Datapgf[[col]])), Datapgf[[col]], linewidth = linewidth)

                # if SET_PARAMS.Fault_names_values[index] == "None":
                #     plt.title(col + " of Perfectly Designed Satellite", fontsize = int(width*1.2))
                # elif recovery == "None":
                #     plt.title(col + " of Without Recovery", fontsize = int(width*1.2))

                if grid:
                    plt.grid(visible = True, which = 'both')

                plt.xlabel("Time: (s)", fontsize = int(width))

                if "Metric" in col:
                    plt.ylabel("$\\theta$ (deg)", fontsize = int(width))
                else:
                    plt.ylabel("Accuracy", fontsize = int(width))

                #for sun in SunInView:
                    # plt.axvline(x=sun, linestyle = '--', c = 'r', linewidth=0.4)

                for eclipse in Eclipse:
                    to = SunInView[next(x[0] for x in enumerate(SunInView) if x[1] > eclipse)]
                    plt.axvspan(eclipse, to , facecolor='grey', alpha=0.2)
                    # plt.axvline(x=eclipse, linestyle = '--', c = 'k', linewidth=0.4)

                plt.tight_layout()

                path = path_of_execution + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"-" + SET_PARAMS.Model_or_Measured +"-General CubeSat Model/" + SET_PARAMS.Fault_names_values[index]
                
                if predictionBuffer:
                    path = path + "BufferValue-" + str(bufferValue) + "BufferStep-" + str(bufferStep) + str(recoveryBuffer) + "/"

                if perfectNoFailurePrediction:
                    path = path + "PerfectNoFailurePrediction/"

                print(path)

                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + col + '.pgf'))

                plt.close()