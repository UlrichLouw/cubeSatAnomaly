from Fault_prediction.Fault_utils import Dataset_order
from sklearn.metrics import confusion_matrix
import numpy as np
from Simulation.Parameters import SET_PARAMS
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
import pickle

def NB(path, featureExtractionMethod, NBType = ["Guassian"], multi_class = False, constellation = False, lowPredictionAccuracy = False, MovingAverage = True, includeAngularMomemntumSensors = False, includeModelled = False):
    X_list = []
    Y_list = []

    pathFiles = SET_PARAMS.path

    buffer = False

    SET_PARAMS.buffer_size = 100

    if constellation:
        for satNum in range(SET_PARAMS.Number_of_satellites):
            print(satNum)
            SET_PARAMS.path = pathFiles + str(satNum) + "/"
            for index in range(SET_PARAMS.number_of_faults):
                name = SET_PARAMS.Fault_names_values[index+1]
                if multi_class:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, constellation = constellation, multi_class = True, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors, includeModelled = includeModelled)
                else:
                    Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, constellation = constellation, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors, includeModelled = includeModelled)
                X_list.append(X)    
                Y_list.append(Y)

    else:
        for index in range(SET_PARAMS.number_of_faults):
            name = SET_PARAMS.Fault_names_values[index+1]
            if multi_class:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = False, categorical_num = True, buffer = buffer, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors, includeModelled = includeModelled)
            else:
                Y, _, X, _, _, ColumnNames, ClassNames = Dataset_order(name, binary_set = True, buffer = buffer, categorical_num = False, MovingAverage = MovingAverage, includeAngularMomemntumSensors = includeAngularMomemntumSensors, includeModelled = includeModelled)
            X_list.append(X)    
            Y_list.append(Y)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)

    Y = Y.reshape(Y.shape[0],)

    # Split data into training and testing data
    mask = np.random.rand(len(X)) <= 0.6
    training_data = X[mask]
    testing_data = X[~mask]

    training_Y = Y[mask]
    testing_Y = Y[~mask]

    # MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
    for nb in NBType:
        print(nb)

        if nb == "Gaussian":
            clf = GaussianNB()
        elif nb == "Multinomial":
            clf = MultinomialNB()
        elif nb == "Complement":
            clf = ComplementNB()
        elif nb == "Bernoulli":
            clf = BernoulliNB()
        elif nb == "Categorical":
            clf = CategoricalNB()

        clf.fit(training_data,training_Y)

        y_pred = clf.predict(testing_data)

        cm = confusion_matrix(testing_Y, y_pred)

        print(cm)

        if multi_class:
            pickle.dump(clf, open(path + '/NaiveBayes' + nb + 'MultiClass.sav', 'wb'))
        else:
            pickle.dump(clf, open(path + '/NaiveBayes' + nb + 'BinaryClass.sav', 'wb'))