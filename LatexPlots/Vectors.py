import matplotlib
import matplotlib.pyplot as plt
from Simulation.Parameters import SET_PARAMS
import pandas as pd
from pathlib import Path
from Extra.util import createParams
import numpy as np

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

def VectorPlots(BufferValue, BufferStep, RecoveryBuffer, PredictionBuffer, PerfectNoFailurePrediction, bbox_to_anchor, loc, featureExtractionMethods, predictionMethods, isolationMethods, recoveryMethods, recoverMethodsWithoutPrediction,index, Number, Number_of_orbits, ALL = True, first = False, width = 8.0, height = 6.0, plotColumns = ["Sun"], singleValue = ["LOF"]):
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
        recoveryMethods = ["None"]

    # plotColumns = ["SolarPanelDipole Torques", "Magnetometer"]
    # plotColumns = ["Estimation Metric"]

    path_of_execution = str(Path(__file__).parent.resolve()).split("/cubeSatAnomaly")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Figures/TexFigures"

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
            
            path = "Data files/"+ GenericPath + SET_PARAMS.Fault_names_values[index] + ".csv.gz"
            path = Path(path)
            Datapgf = GetData(path, index, n = Number, all = ALL, first = first) 
            
            currenctSunStatus = Datapgf.loc[0, "Sun in view"]
            currentAnomalyStatus = 0

            SunInView = []
            Eclipse = []
            Anomaly = []
            Normal = []
            anomalySpan = []
            eclipseSpan = []
            
            for ind in Datapgf.index:
                if Datapgf.loc[ind, "Sun in view"] != currenctSunStatus:
                    if Datapgf.loc[ind, "Sun in view"]:
                        SunInView.append(ind)
                        eclipseSpan.append([Eclipse[-1], ind - 1])
                    else:
                        Eclipse.append(ind)
                
                if Datapgf.loc[ind, "Current fault binary"] != currentAnomalyStatus:
                    if Datapgf.loc[ind, "Current fault binary"] == 1:
                        Anomaly.append(ind)
                    else:
                        Normal.append(ind)
                        anomalySpan.append([Anomaly[-1], ind - 1])

                currenctSunStatus = Datapgf.loc[ind, "Sun in view"]
                currentAnomalyStatus = Datapgf.loc[ind, "Current fault binary"]

            if Datapgf.loc[ind, "Current fault binary"] == 1:
                anomalySpan.append([Anomaly[-1], ind])

            if not Datapgf.loc[ind, "Sun in view"]:
                eclipseSpan.append([Eclipse[-1], ind])

            for col in plotColumns:
                if col in singleValue:
                    if col == 'LOF':
                        Datapgf['LOF'] = Datapgf['LOF'].str.replace(r'\[', '', regex=True)
                        Datapgf['LOF'] = Datapgf['LOF'].str.replace(r'\]', '', regex=True).apply(np.float32)
                        Datapgf['LOF'] = 1/Datapgf['LOF'] * (-1)

                    if col == 'DMD':
                        Datapgf['Moving Average'] = Datapgf['Moving Average'].apply(lambda x: 
                           sum(np.absolute(np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))))
                        # for ind in Datapgf.index:
                        #     Datapgf['Moving Average'] = np.sum(Datapgf.loc[ind, 'Moving Average'])
                        
                        Datapgf['DMD'] = Datapgf['Moving Average'].apply(np.float32)
                    
                    plt.figure(figsize = (width*cm, height*cm))

                    plt.plot(range(len(Datapgf)), Datapgf[[col]], alpha=0.7, linewidth=1, color=(0,0,0.5))

                    plt.grid(visible = True, which = 'both')

                    plt.xlabel("Time: ($s$)", fontsize = int(width))

                    if 'Torque' in col:
                        plt.ylabel("Torque: ($N \\cdot m$)", fontsize = int(width))

                    # for ind in Datapgf.index:
                    #     if Datapgf.loc[ind,"Current fault binary"] == 1:
                    #         plt.axvline(x = ind, color='red', alpha=0.2)

                    for anomaly in anomalySpan:
                        # to = Normal[next(x[0] for x in enumerate(Normal) if x[1] > anomaly)]
                        plt.axvspan(anomaly[0], anomaly[1] , facecolor='red', alpha=0.2)

                    # plt.legend(['$\\bar{\mathbf{x}}_\mathcal{B}$'], loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                    plt.tight_layout()

                    path = path_of_execution + "/FeatureExtraction-" + SET_PARAMS.FeatureExtraction +  "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"-" + SET_PARAMS.Model_or_Measured +"-General CubeSat Model/" + SET_PARAMS.Fault_names_values[index]
                    
                    Path(path).mkdir(parents = True, exist_ok=True)

                    plt.savefig(Path(path + "/" + col + '.pgf'))

                    plt.close()
                else:
                    plt.figure(figsize = (width*cm, height*cm))

                    plt.plot(range(len(Datapgf)), Datapgf[[col+ "_x"]], alpha=0.7, linewidth=1, color=(0.5,0,0))

                    plt.plot(range(len(Datapgf)), Datapgf[[col+ "_y"]], alpha=0.7, linewidth=1, color=(0,0.5,0))

                    plt.plot(range(len(Datapgf)), Datapgf[[col+ "_z"]], alpha=0.7, linewidth=1, color=(0,0,0.5))

                    plt.grid(visible = True, which = 'both')

                    plt.xlabel("Time: ($s$)", fontsize = int(width))

                    if 'Torque' in col:
                        plt.ylabel("Torque: ($N \\cdot m$)", fontsize = int(width))


                    for eclipse in Eclipse:
                        to = SunInView[next(x[0] for x in enumerate(SunInView) if x[1] > eclipse)]
                        plt.axvspan(eclipse, to , facecolor='grey', alpha=0.2)

                    plt.legend(['$\\bar{\mathbf{x}}_\mathcal{B}$', '$\\bar{\mathbf{y}}_\mathcal{B}$', '$\\bar{\mathbf{z}}_\mathcal{B}$'], loc = loc, fontsize = int(width), bbox_to_anchor=bbox_to_anchor, handlelength = 0.2)

                    plt.tight_layout()

                    path = path_of_execution + "/FeatureExtraction-" + SET_PARAMS.FeatureExtraction + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"-" + SET_PARAMS.Model_or_Measured +"-General CubeSat Model/" + SET_PARAMS.Fault_names_values[index]
                    
                    Path(path).mkdir(parents = True, exist_ok=True)

                    plt.savefig(Path(path + "/" + col + '.pgf'))

                    plt.close()