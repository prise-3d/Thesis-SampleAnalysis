# main imports
import os, sys, argparse
import numpy as np
import json
import matplotlib.pyplot as plt

# model imports
from joblib import dump
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg


def train(_data_file, _model_name):

    # get length of data
    dataset=np.loadtxt(_data_file, delimiter=';')

    y = dataset[:,0]
    X = dataset[:,1:]
    _, nb_elem = X.shape

    y=np.reshape(y, (-1,1))
    scaler = MinMaxScaler()

    scaler.fit(X)
    scaler.fit(y)
    
    xscale=scaler.transform(X)
    yscale=scaler.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

    # define keras NN structure
    model = Sequential()
    model.add(Dense(200, input_dim=nb_elem, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.summary()

    # Set expected metrics
    # TODO : add coefficients of determination as metric ? Or always use MSE/MAE
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    history = model.fit(X_train, y_train, epochs=cfg.keras_epochs, batch_size=50,  verbose=1, validation_split=0.2)

    # save the model into json/HDF5 file
    if not os.path.exists(cfg.saved_models_folder):
        os.makedirs(cfg.saved_models_folder)

    model_output_path = os.path.join(cfg.saved_models_folder, _model_name + '.json')
    json_model_content = model.to_json()

    with open(model_output_path, 'w') as f:
        print("Model saved into ", model_output_path)
        json.dump(json_model_content, f, indent=4)

    model.save_weights(model_output_path.replace('.json', '.h5'))

    # save score into global_result.csv file
    # if not os.path.exists(cfg.results_information_folder):
    #    os.makedirs(cfg.results_information_folder)
    #
    # with open(cfg.global_result_filepath, "a") as f:
    #   f.write(_model_name + ';' + str(len(y)) + ';' + str(coeff[0]) + ';\n')


    # Save plot info using model name
    plt.figure(figsize=(30, 22))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss', fontsize=20)
    plt.ylabel('loss', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.legend(['train', 'validation'], loc='upper left', fontsize=16)
    plt.savefig(model_output_path.replace('.json', '.png'))


def main():

    parser = argparse.ArgumentParser(description="Train model and saved it")

    parser.add_argument('--data', type=str, help='Filename of dataset')
    parser.add_argument('--model_name', type=str, help='Saved model name')

    args = parser.parse_args()

    param_data_file = args.data
    param_model = args.model_name

    train(param_data_file, param_model)

if __name__== "__main__":
    main()