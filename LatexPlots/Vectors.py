import matplotlib
import matplotlib.pyplot as plt
from Simulation.Parameters import SET_PARAMS
import pandas as pd
from pathlib import Path
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

def VectorPlots(BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction, bbox_to_anchor, loc, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction,index, Number, Number_of_orbits, ALL = True, first = False, width = 8.0, height = 6.0):
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.Model_or_Measured = "ORC"
    SET_PARAMS.Number_of_orbits = Number_of_orbits
    SET_PARAMS.save_as = ".csv"
    SET_PARAMS.Low_Aerodynamic_Disturbance = False

    cm = 1/2.54


    if index == 1:
        isolationMethods = ["None"]
        recoveryMethods = ["None"]
        recoverMethodsWithoutPrediction = ["None"]

    plotColumns = ["Sun"]
    # plotColumns = ["Estimation Metric"]

    path_of_execution = str(Path(__file__).parent.resolve()).split("/Satellite")[0] + "/Journal articles/My journal articles/Journal articles/Robust Kalman Filter/Figures/TexFigures"

    Path(path_of_execution).mkdir(parents = True, exist_ok=True)

    satelliteFDIRParams = createParams(PredictionBuffer, BufferValue, BufferStep, PerfectNoFailurePrediction, RecoveryBuffer, featureExtractionMethods, recoveryMethods, predictionMethods, isolationMethods)


    for predictionBuffer, bufferValue, bufferStep, perfectNoFailurePrediction, recoveryBuffer, extraction, recovery, prediction, isolation in satelliteFDIRParams:
        if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None") or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
            SET_PARAMS.FeatureExtraction = extraction
            SET_PARAMS.SensorPredictor = prediction
            SET_PARAMS.SensorIsolator = isolation
            SET_PARAMS.SensorRecoveror = recovery
            GenericPath = "FeatureExtraction-" + SET_PARAMS.FeatureExtraction + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/"+ "General CubeSat Model/"

            if SET_PARAMS.Low_Aerodynamic_Disturbance:
                GenericPath = "Low_Disturbance/" + GenericPath

            if predictionBuffer:
                GenericPath = GenericPath + "BufferValue-" + str(bufferValue) + "BufferStep-" + str(bufferStep) + recoveryBuffer + "/"

            if perfectNoFailurePrediction:
                GenericPath = GenericPath + "PerfectNoFailurePrediction/"
            
            path = "Data files/"+ GenericPath + SET_PARAMS.Fault_names_values[index] + ".csv"
            path = Path(path)
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

                plt.plot(range(len(Datapgf)), Datapgf[[col+ "_x"]], alpha=0.7, linewidth=1)

                plt.plot(range(len(Datapgf)), Datapgf[[col+ "_y"]], alpha=0.7, linewidth=1)

                plt.plot(range(len(Datapgf)), Datapgf[[col+ "_z"]], alpha=0.7, linewidth=1)

                plt.grid(visible = True, which = 'both')

                plt.xlabel("Time: (s)", fontsize = int(width))


                for eclipse in Eclipse:
                    to = SunInView[next(x[0] for x in enumerate(SunInView) if x[1] > eclipse)]
                    plt.axvspan(eclipse, to , facecolor='grey', alpha=0.2)

                plt.legend(['x', 'y', 'z'], loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                plt.tight_layout()

                path = path_of_execution + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"-" + SET_PARAMS.Model_or_Measured +"-General CubeSat Model/" + SET_PARAMS.Fault_names_values[index]
                
                Path(path).mkdir(parents = True, exist_ok=True)

                plt.savefig(Path(path + "/" + col + '.pgf'))

                plt.close()