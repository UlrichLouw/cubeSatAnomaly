import pandas as pd
from Simulation.Parameters import SET_PARAMS
from pathlib import Path
from Simulation.Save_display import visualize_data, save_as_csv, save_as_pickle
import csv
import os
import glob
import numpy as np
import sklearn.metrics as metrics
from Extra.util import createParams

def GetData(path, nameList):
    ###################################################################################################################################################################################
    # This is the globalVariables.pathOfWorkbook to the folder for the archived circuitDB
    excel_folder = os.path.join(path)
    ####################################################################################################################################################################################
    # glob.glob returns all files matching the pattern.
    excel_files = list(glob.glob(os.path.join(excel_folder, '*.csv.gz*')))
    ####################################################################################################################################################################################
    # Append all the csv files in the Dataframe list 
    Dataframe = []

    for f in excel_files:
        df = pd.read_csv(f, engine='c')
        df = df[nameList]
        Dataframe.append(df)
    ####################################################
    #     IF THERE IS NO EXISTING CSV FILES AND NO     #
    # PREVIOUS VERSIONS IGNORE IMPORT PREVIOUS VERSION #
    ####################################################
    
    return Dataframe

def dataFrameToLatex():
    pass

def SaveSummary(path, method, recovery, prediction, col, getData = True, DataFrame = None, specific = False):

    if col == "Prediction Accuracy":
        colCollect = ["Prediction Accuracy", "Predicted fault", "Current fault binary"]
        
    else:
        colCollect = [col]

    if getData and not specific:
        DataFrames = GetData(path, colCollect)
    elif getData:
        DataFrames = [pd.read_csv(path, low_memory = False, engine = 'c')]
    else:
        DataFrames = [DataFrame]
    
    meanList, stdList, columns = [], [], []

    columns.append(("Orbits", "Detection Strategy", "Detection Strategy"))
    columns.append(("Orbits", "Recovery Strategy", "Recovery Strategy"))

    for num in range(1,SET_PARAMS.Number_of_orbits+1):
        columns.append((num, "Metric ($\\theta$)", 'Mean'))
        columns.append((num, "Metric ($\\theta$)", 'Std'))

    columns = pd.MultiIndex.from_tuples(columns)

    df = pd.DataFrame(columns = columns, index = [method])

    df.loc[method, ("Orbits", "Detection Strategy", "Detection Strategy")] = prediction

    df.loc[method, ("Orbits", "Recovery Strategy", "Recovery Strategy")] = recovery
    
    if col == "Prediction Accuracy":
        cm = metrics.confusion_matrix(DataFrames[0]["Current fault binary"], DataFrames[0]["Predicted fault"])
    else:
        cm = np.array(([None]))

    for orbit in range(1,SET_PARAMS.Number_of_orbits+1):
        for DataFrame in DataFrames:
            DF = DataFrame[int((orbit-1)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts)):int((orbit)*SET_PARAMS.Period/(SET_PARAMS.faster_than_control*SET_PARAMS.Ts))]
                        
            Metric = DF[col]

            meanList.append(Metric.mean())
            stdList.append(Metric.std())

        df.loc[method, (orbit,"Metric ($\\theta$)",'Mean')] = sum(meanList)/len(meanList)
        df.loc[method, (orbit,"Metric ($\\theta$)",'Std')] = sum(stdList)/len(stdList)

    return df, cm


def multiIndexToLatex(df, headers, columsPosition = "c", tablePosition = "[]", caption = "RandomTable", label = "RandomLabel", tableDoubleColumn = False, highlightMax = True, highlightMin = False, levelsOfHeadersToHighlight = [-1], cm = True):
    
    # Create a single or double column table for articles
    if tableDoubleColumn:
        string = "\\begin{table*}" + tablePosition + " \n" 
    else:
        string = "\\begin{table}" + tablePosition + " \n"

    # Generate caption
    string += "\caption{" + caption + "} \n"

    # Generate label
    string += "\label{" + label + "} \n"

    # center the table
    string += "\centering \n"

    string += "\\begin{tabular} \n {@{}"

    # Generate the format for columns
    string += (len(headers[0]) + 1) * columsPosition + "@{}} \n"

    # Insert top rule
    string += "\\toprule \n"

    for i in range(len(headers)):
        colLevels = headers[i]
        cmidrule = False
        for ind in range(len(colLevels)):
            if (colLevels[ind] != colLevels[ind - 1] or ind == 0) and (headers[i][ind] != headers[i-1][ind] or i == 0):
                string += "\multicolumn{"
                
                # Calculate the number of columns that must be merged in the heading
                numberOfColumns = 1
                for num in range(ind + 1, len(colLevels)):
                    if colLevels[num] == colLevels[ind]:
                        numberOfColumns += 1
                    else:
                        break
                        
                # borders on left and right hand side
                if i > 0 and ind == 0:
                    string += str(numberOfColumns) + "}{|" + columsPosition + "|}"
                elif i > 0:
                    string += str(numberOfColumns) + "}{" + columsPosition + "|}"
                else:
                    string += str(numberOfColumns) + "}{" + columsPosition + "}"

                # Calculate the number of columns that must be merged in the heading
                numberOfRows = 1
                for num in range(i + 1, len(headers)):
                    if headers[i][ind] == headers[i+1][ind]:
                        numberOfRows += 1
                    else:
                        break
                
                if numberOfRows > 1:
                    cmidrule = True
                    startCol = ind + 2
                    string += "{\multirow{" + str(numberOfRows) + "}{*}"

                    if i in levelsOfHeadersToHighlight:
                        string += "{\\textbf{" + colLevels[ind] + "}}}"
                    else:
                        string += "{" + colLevels[ind] + "}}"
                else:
                    if i in levelsOfHeadersToHighlight:
                        string += "{\\textbf{" + colLevels[ind] + "}}"
                    else:
                        string += "{" + colLevels[ind] + "}"

                if ind < len(colLevels) - numberOfColumns:
                    string += " & \n"

            elif headers[i][ind] == headers[i-1][ind]:
                string += "\multicolumn{1}{"  

                # borders on left and right hand side
                if ind == 0:
                    string += "|" + columsPosition + "|}{}"
                else:
                    string += columsPosition + "|}{}"
            
                if ind < len(colLevels) - 1:
                    string += " & \n"
            
            
        if not cmidrule:
            string += "\n \\\ \midrule \n" 
        else:
            string += "\n \\\ \cmidrule(l){" + str(startCol) + "-" + str(len(colLevels)) + "} \n"  

    # Insert the data from the dataframe after the columns
    entries = df.values

    array = np.array(entries)

    if highlightMax:
        maximums = np.amax(array, axis = 0)
    else:
        maximums = ["No value"]
    
    if highlightMin:
        minimums = np.min(array, axis = 0)
    else:
        minimums = ["No value"]

    numberOfcmidrules = 0

    for i in range(len(entries)):
        values = entries[i]

        for ind in range(len(values)):
            if (values[ind] != values[ind - 1] or not isinstance(entries[i][ind], str) or ind == 0) and (entries[i][ind] != entries[i-1][ind] or not isinstance(entries[i][ind], str) or i == 0):          
                string += "\multicolumn{1}{"   

                # Calculate the number of columns that must be merged in the heading
                numberOfRows = 1

                # borders on left and right hand side
                if ind == 0:
                    string += "|" + columsPosition + "|}"
                else:
                    string += columsPosition + "|}"

                for num in range(i + 1, len(entries)):
                    if entries[i][ind] == entries[num][ind] and isinstance(entries[i][ind],str):
                        numberOfRows += 1
                    else:
                        break
                
                if numberOfRows > 1:
                    numberOfcmidrules = numberOfRows
                    string += "{\multirow{" + str(numberOfRows) + "}{*}"
                    if isinstance(values[ind], str):
                        string += "{" + values[ind] + "}}"
                    else:
                        if cm and values[ind] in maximums:
                            string += "{\color{green}\\textbf{" + str(values[ind]) + "}}}"
                        elif cm:
                            string += "{\color{red}\\textbf{" + str(values[ind]) + "}}}"

                        elif values[ind] in maximums or values[ind] in minimums:
                            string += "{\\textbf{" + "{:0.2f}".format(values[ind]) + "}}}"
                        else:
                            string += "{" + "{:0.2f}".format(values[ind]) + "}}"
                    
                else:
                    if isinstance(values[ind], str):
                        string += "{" + values[ind] + "}"
                    else:
                        if cm and values[ind] in maximums:
                            string += "{\color{green}\\textbf{" + str(values[ind]) + "}}"
                        elif cm:
                            string += "{\color{red}\\textbf{" + str(values[ind]) + "}}"

                        elif values[ind] in maximums or values[ind] in minimums:
                            string += "{\\textbf{" + "{:0.2f}".format(values[ind]) + "}}"
                        else:
                            string += "{" + "{:0.2f}".format(values[ind]) + "}"

            elif entries[i][ind] == entries[i-1][ind]:
                string += "\multicolumn{1}{"  
               
                # borders on left and right hand side
                if ind == 0:
                    string += "|" + columsPosition + "|}{}"
                else:
                    string += columsPosition + "|}{}"

            elif values[ind] == values[ind - 1]:
                string += "\multicolumn{1}{"  
               
                # borders on left and right hand side
                if ind == 0:
                    string += "|" + columsPosition + "|}"
                else:
                    string += columsPosition + "|}"

                if isinstance(values[ind], str):
                    string += "{" + values[ind] + "}"
                else:
                    string += "{" + "{:0.2f}".format(values[ind]) + "}"

            if ind < len(values) - 1:
                string += " & \n"
            elif i == len(entries) - 1:
                string += "\n \\\ \\bottomrule \n"
            elif numberOfcmidrules <= 1:
                string += "\n \\\ \midrule \n" 
            else:
                string += "\n \\\ \cmidrule(l){2-" + str(len(values)) + "} \n"  
                numberOfcmidrules -= 1


    string += "\end{tabular} \n"

    if tableDoubleColumn:
        string += "\\end{table*} \n" 
    else:
        string += "\\end{table} \n"

    return string

if __name__ == "__main__":
    doNotOverwriteSummary = False
    featureExtractionMethods = ["None", "LOF", "DMD"]
    treeDepth = [100] #[5, 10, 20, 100]
    predictionMethod = [100.0, "DecisionTrees", "RandomForest", "SVM", "IsolationForest", "LOF", 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 99.0] #, "NaiveBayesBernoulli", "NaiveBayesGaussian", , 50.0, 60.0, 70.0, 80.0, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5, 99.9] #   
    isolationMethod = [100.0, "DecisionTrees", "RandomForest", "SVM", 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 99.0]
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

     #! "RandomForest", 
    recoveryMethods = ["EKF-ignore", "EKF-combination", "EKF-reset"] #, "EKF-top2"] # 
    recoverMethodsWithoutPrediction = ["None", "EKF-top2"]
    # predictionMethods = ["DecisionTrees"]
    # isolationMethods = ["RandomForest"] #! "RandomForest", 
    # recoveryMethods = ["EKF-replacement"]
    SET_PARAMS.Mode = "EARTH_SUN"
    SET_PARAMS.Model_or_Measured = "ORC"
    SET_PARAMS.Number_of_orbits = 30

    SET_PARAMS.RecoveryBuffer = ["EKF-top2"]
    SET_PARAMS.PredictionBuffer = [False] #, True]
    SET_PARAMS.BufferValue = [10]
    SET_PARAMS.BufferStep = [0.9]
    SET_PARAMS.perfectNoFailurePrediction =[False] #, True]

    index = 2

    includeNone = True

    nameList = ["Isolation Accuracy", "Prediction Accuracy", "Pointing Metric", "Estimation Metric"]

    orbitsToLatex = [1, 2, 3, 4, 5, 30]

    nameDict = {x: [] for x in nameList}

    summaryDict = {x: 0 for x in nameList}

    for index in range(2,4):
        for name in nameList:
            if doNotOverwriteSummary:
                path = "Data files/Summary/" + name + "/" + SET_PARAMS.Fault_names_values[index] + ".csv"

                prevSummary = pd.read_csv(path)

                summaryDict[name] = prevSummary



        # with open(filename, 'w') as csvfile:
        # creating a csv writer object
            # csvwriter = csv.writer(csvfile)

        pathParams = createParams(SET_PARAMS.PredictionBuffer, SET_PARAMS.perfectNoFailurePrediction, SET_PARAMS.BufferValue, SET_PARAMS.BufferStep, SET_PARAMS.RecoveryBuffer)

        for predictionBuffer, perfectNoFailurePrediction, bufferValue, bufferStep, recoveryBuffer in pathParams:

            path_of_execution = str(Path(__file__).parent.resolve()).split("/Satellite")[0] + "/stellenbosch_ee_report_template-master/Masters Thesis/Tables/" + name

            if predictionBuffer:
                path_of_execution = path_of_execution + "BufferValue-" + str(bufferValue) + "BufferStep-" + str(bufferStep) + recoveryBuffer + "/"

            if perfectNoFailurePrediction:
                path_of_execution = path_of_execution + "PerfectNoFailurePrediction/"

            Path(path_of_execution).mkdir(parents = True, exist_ok=True)

            satelliteFDIRParams = createParams(featureExtractionMethods, recoveryMethods, predictionMethods, isolationMethods)
                
            for extraction, recovery, prediction, isolation in satelliteFDIRParams:
                if (recovery in recoverMethodsWithoutPrediction and prediction == "None" and isolation == "None" and not predictionBuffer and not perfectNoFailurePrediction) or (prediction != "None" and isolation != "None" and recovery not in recoverMethodsWithoutPrediction):
                    SET_PARAMS.FeatureExtraction = extraction
                    SET_PARAMS.SensorPredictor = str(prediction)
                    SET_PARAMS.SensorIsolator = str(isolation)
                    SET_PARAMS.SensorRecoveror = recovery
                    GenericPath = "FeatureExtraction-" + str(SET_PARAMS.FeatureExtraction) + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/" + SET_PARAMS.Model_or_Measured +"/" + "General CubeSat Model/"
                    method = extraction + str(prediction) + str(isolation) + recovery + SET_PARAMS.Fault_names_values[index] 

                    if predictionBuffer:
                        GenericPath = GenericPath + "BufferValue-" + str(bufferValue) + "BufferStep-" + str(bufferStep) + recoveryBuffer + "/"
                        method += str(bufferValue) + str(bufferStep) + recoveryBuffer

                    if perfectNoFailurePrediction:
                        GenericPath = GenericPath + "PerfectNoFailurePrediction/"
                        method += str("perfectNoFailurePrediction")

                    path1 = "Data files/"+ GenericPath + SET_PARAMS.Fault_names_values[index] 
                    
                    execute = True
                    print("Begin: " + method)
                    if doNotOverwriteSummary:
                        if method in prevSummary["Unnamed: 0"]:
                            execute =  False
                    
                    if execute:
                        path = Path(path1 + ".csv.gz")

                        NoDataFrame = False
                        try:
                            df = pd.read_csv(path, engine='c')
                        except:
                            df = None

                        for name in nameList:
                            try:
                                dataFrame, cm = SaveSummary(path, method, str(recovery), str(prediction), name, DataFrame = df, specific = True)
                            except:
                                NoDataFrame = True 

                            if not NoDataFrame:
                                # writing the fields
                                # csvwriter.writerow(dataFrame)
                                nameDict[name].append(dataFrame.copy())

                                if (cm != None).any():
                                    df = pd.DataFrame(cm)  
                                    headers = [
                                    ["Predicted", "Predicted"],
                                    ["Failure", "No Failure"]]
                                    highlightMax = True
                                    highlightMin = False
                                    string = multiIndexToLatex(df, headers, columsPosition = "c", tablePosition = "[]", caption = "Confusion Matric for " + str(prediction), label = "Table: " + name + "-" + method + "-" + SET_PARAMS.Fault_names_values[index], tableDoubleColumn = False, highlightMax = highlightMax, highlightMin = highlightMin, levelsOfHeadersToHighlight = [0], cm = True)
                                    f = open(Path(path_of_execution + "/" + name + "-" + method + "-" + SET_PARAMS.Fault_names_values[index] + ".tex"),"w")

                                    f.write(string)

        for name in nameList:             
            # SET_PARAMS.FeatureExtraction = "DMD"
            # SET_PARAMS.SensorPredictor = "None"
            # SET_PARAMS.SensorIsolator = "None"
            # SET_PARAMS.SensorRecoveror = "None"
            # GenericPath = "FeatureExtraction-" + str(SET_PARAMS.FeatureExtraction) + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+SET_PARAMS.Model_or_Measured +"/" + "General CubeSat Model/"
            # path = "Data files/"+ GenericPath + "/" + SET_PARAMS.Fault_names_values[index]
            # method = "DMD" + "None" + "None" + "None" + SET_PARAMS.Fault_names_values[index]
            # print("Begin: " + method)
            # path = Path(path + ".csv.gz")
            # dataFrame, cm = SaveSummary(path, method, "Failure Design", "None", name, getData = False, DataFrame = pd.read_csv(path, engine='c'))
            # nameDict[name].append(dataFrame.copy())
            # csvwriter.writerow(dataFrame)

            if includeNone:
                SET_PARAMS.FeatureExtraction = "None"
                SET_PARAMS.SensorPredictor = "None"
                SET_PARAMS.SensorIsolator = "None"
                SET_PARAMS.SensorRecoveror = "None"
                GenericPath = "FeatureExtraction-" + str(SET_PARAMS.FeatureExtraction) + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+SET_PARAMS.Model_or_Measured +"/" + "General CubeSat Model/"
                path = "Data files/"+ GenericPath + "/" + SET_PARAMS.Fault_names_values[index] 
                method = "None" + "None" + "None" + "None" + SET_PARAMS.Fault_names_values[index] 
                print("Begin: " + method)
                path = Path(path + ".csv.gz")
                dataFrame, cm = SaveSummary(path, method, "Perfect Design", "None", name, getData = False, DataFrame = pd.read_csv(path, engine='c'))
                nameDict[name].append(dataFrame.copy())
                # csvwriter.writerow(dataFrame)

            if includeNone:
                SET_PARAMS.FeatureExtraction = "None"
                SET_PARAMS.SensorPredictor = "None"
                SET_PARAMS.SensorIsolator = "None"
                SET_PARAMS.SensorRecoveror = "None"
                GenericPath = "FeatureExtraction-" + str(SET_PARAMS.FeatureExtraction) + "/Predictor-" + SET_PARAMS.SensorPredictor+ "/Isolator-" + SET_PARAMS.SensorIsolator + "/Recovery-" + SET_PARAMS.SensorRecoveror +"/"+SET_PARAMS.Mode+"/"+SET_PARAMS.Model_or_Measured +"/" + "General CubeSat Model/"
                path = "Data files/"+ GenericPath + "/" + "None"
                method = "None" + "None" + "None" + "None" + "None"
                print("Begin: " + method)
                path = Path(path + ".csv.gz")
                dataFrame, cm = SaveSummary(path, method, "Perfect Design", "None", name, getData = False, DataFrame = pd.read_csv(path, engine='c'))
                nameDict[name].append(dataFrame.copy())

        # save_as_csv(dataFrame, filename = SET_PARAMS.Fault_names_values[index], index = index, path = path,  float_format="%.2f")

        for name in nameList:
            dataFrame = pd.concat(nameDict[name])
            path = "Data files/Summary/" + name + "/"

            path_to_folder = Path(path)
            path_to_folder.mkdir(parents = True, exist_ok=True)

            filename = path + SET_PARAMS.Fault_names_values[index] + ".csv"

            if os.path.exists(filename) and not doNotOverwriteSummary:
                os.remove(filename)

            dataFrame.to_csv(filename)

            for orbit in range(1,SET_PARAMS.Number_of_orbits+1):
                if orbit not in orbitsToLatex:
                    dataFrame = dataFrame.loc[:, dataFrame.columns != (orbit,"Metric ($\\theta$)",'Mean')]
                    dataFrame = dataFrame.loc[:, dataFrame.columns != (orbit,"Metric ($\\theta$)",'Std')]

            dataFrame = dataFrame.reset_index(drop = True)

            # dataFrame.fillna("", inplace = True)

            headers = [
                ["Orbits", "Orbits", "1", "1", "2", "2", "3", "3", "4", "4", "5", "5", "30", "30"],
                ["Detection Strategy", "Recovery Strategy", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)", "Metric ($\\theta$)"],
                ["Detection Strategy", "Recovery Strategy", "Mean", "Std", "Mean", "Std", "Mean", "Std", "Mean", "Std", "Mean", "Std", "Mean", "Std"]
            ]

            dataFrame.columns = headers

            if name == "Prediction Accuracy" or name == "Isolation Accuracy":
                highlightMax = True
                highlightMin = False
            else:
                highlightMax = False
                highlightMin = True

            string = multiIndexToLatex(dataFrame, headers, columsPosition = "c", tablePosition = "[]", caption = name + " for various methods", label = "Table: " + name + "-" + SET_PARAMS.Fault_names_values[index], tableDoubleColumn = True, highlightMax = highlightMax, highlightMin = highlightMin, levelsOfHeadersToHighlight = [0])

            f = open(Path(path_of_execution + "/" + SET_PARAMS.Fault_names_values[index] + ".tex"),"w")

            f.write(string)