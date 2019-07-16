# main imports
import numpy as np
import pandas as pd
import os, sys, argparse

# model imports
from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle
from joblib import dump, load

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from ipfml import metrics


def get_model_choice(_model_name):
    """
    Bind choose model using String information
    """

    if _model_name == "SGD":
        clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)

    if _model_name == "Ridge":
        clf = linear_model.Ridge(alpha=1.)

    if _model_name == "SVR":
        clf = svm.SVR()

    return clf

def train(_data_file, _model_name):

    # prepare data
    dataset = pd.read_csv(_data_file, header=None, sep=";")
    dataset = shuffle(dataset)

    y = dataset.ix[:,0]
    X = dataset.ix[:,1:]

    clf = get_model_choice(_model_name)
    clf.fit(X, y)

    y_predicted = clf.predict(X)

    coeff = metrics.coefficient_of_determination(y, y_predicted)

    print("Predicted coefficient of determination for ", _model_name, " : ", coeff)

    # save the trained model, so check if saved folder exists
    if not os.path.exists(cfg.saved_models_folder):
        os.makedirs(cfg.saved_models_folder)

    # compute model filename_colum,n
    model_filename = _data_file.split('/')[-1].replace(cfg.output_file_prefix, '').replace('.csv', '')
    model_filename = model_filename + '_' + _model_name + '.joblib'

    model_file_path = os.path.join(cfg.saved_models_folder, model_filename)
    print("Model will be save into `", model_file_path, '`')

    dump(clf, model_file_path)

    # save score into global_result.csv file
    with open(cfg.global_result_filepath, "a") as f:
       f.write(model_filename.replace('.joblib', '') + ';' + str(len(y)) + ';' + str(coeff) + ';\n')

def main():

    parser = argparse.ArgumentParser(description="Train model and saved it")

    parser.add_argument('--data', type=str, help='Filename of dataset')
    parser.add_argument('--model', type=str, help='Kind of model expected', choices=cfg.kind_of_models)

    args = parser.parse_args()

    param_data_file = args.data
    param_model = args.model

    train(param_data_file, param_model)

if __name__== "__main__":
    main()
