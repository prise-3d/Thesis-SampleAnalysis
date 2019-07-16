# main imports
import numpy as np
import pandas as pd
import json
import os, sys, argparse, subprocess

# model imports
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

# image processing imports
from PIL import Image
import ipfml.iqa.fr as fr
from ipfml import metrics

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

n_samples_image_name_postfix = "_samples_mean.png"
reference_image_name_postfix = "_1000_samples_mean.png"

def write_result(_scene_name, _data_file, _model_path, _n, _reconstructed_path, _iqa):
    
    # prepare data to get score information
    dataset=np.loadtxt(_data_file, delimiter=';')

    y = dataset[:,0]
    X = dataset[:,1:]

    y=np.reshape(y, (-1,1))
    scaler = MinMaxScaler()

    scaler.fit(X)
    scaler.fit(y)

    xscale=scaler.transform(X)
    yscale=scaler.transform(y)

    _, X_test, _, y_test = train_test_split(xscale, yscale)

    # prepare image path to compare
    n_samples_image_path = os.path.join(cfg.reconstructed_folder, _scene_name + '_' + _n + n_samples_image_name_postfix)
    reference_image_path = os.path.join(cfg.reconstructed_folder, _scene_name + reference_image_name_postfix)

    if not os.path.exists(n_samples_image_path):
        # call sub process to create 'n' samples img
        print("Creation of 'n' samples image : ", n_samples_image_path)
        subprocess.run(["python", "reconstruct/reconstruct_scene_mean.py", "--scene", _scene_name, "--n", _n, "--image_name", n_samples_image_path.split('/')[-1]])

    if not os.path.exists(reference_image_path):
        # call sub process to create 'reference' img
        print("Creation of reference image : ", reference_image_path)
        subprocess.run(["python", "reconstruct/reconstruct_scene_mean.py", "--scene", _scene_name, "--n", str(1000), "--image_name", reference_image_path.split('/')[-1]])


    # load the trained model
    with open(_model_path, 'r') as f:
        json_model = json.load(f)
        model = model_from_json(json_model)
        model.load_weights(_model_path.replace('.json', '.h5'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    
    # get coefficient of determination score on test set
    y_predicted = model.predict(X_test)
    len_shape, _ = y_predicted.shape
    y_predicted = y_predicted.reshape(len_shape)

    coeff = metrics.coefficient_of_determination(y_test, y_predicted)

    # Get data information
    reference_image = Image.open(reference_image_path)
    reconstructed_image = Image.open(_reconstructed_path)
    n_samples_image = Image.open(n_samples_image_path)

    # Load expected IQA comparison
    try:
        fr_iqa = getattr(fr, _iqa)
    except AttributeError:
        raise NotImplementedError("FR IQA `{}` not implement `{}`".format(fr.__name__, _iqa))

    mse_ref_reconstructed_samples = fr_iqa(reference_image, reconstructed_image)
    mse_reconstructed_n_samples = fr_iqa(n_samples_image, reconstructed_image)

    model_name = _model_path.split('/')[-1].replace('.json', '')

    if not os.path.exists(cfg.results_information_folder):
        os.makedirs(cfg.results_information_folder)
    
    # save score into models_comparisons_keras.csv file
    with open(cfg.global_result_filepath_keras, "a") as f:
       f.write(model_name + ';' + str(len(y)) + ';' + str(coeff[0]) + ';' + str(mse_reconstructed_n_samples) + ';' + str(mse_ref_reconstructed_samples) + '\n')

def main():

    parser = argparse.ArgumentParser(description="Train model and saved it")

    parser.add_argument('--scene', type=str, help='Scene name to reconstruct', choices=cfg.scenes_list)
    parser.add_argument('--data', type=str, help='Filename of dataset')
    parser.add_argument('--model_path', type=str, help='Json model file path')
    parser.add_argument('--n', type=str, help='Number of pixel values approximated to keep')
    parser.add_argument('--image_path', type=str, help="The image reconstructed to compare with")
    parser.add_argument('--iqa', type=str, help='Image to compare', choices=['ssim', 'mse', 'rmse', 'mae', 'psnr'])
  
    args = parser.parse_args()

    param_scene_name = args.scene
    param_data_file = args.data
    param_n = args.n
    param_model_path = args.model_path
    param_image_path = args.image_path
    param_iqa = args.iqa

    write_result(param_scene_name, param_data_file, param_model_path, param_n, param_image_path, param_iqa)

if __name__== "__main__":
    main()
