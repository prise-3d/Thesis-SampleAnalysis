import numpy as np
import pandas as pd
import json
import os, sys, argparse

from keras.models import model_from_json

import modules.config as cfg
import modules.metrics as metrics

from joblib import dump, load

from PIL import Image

def reconstruct(_scene_name, _model_path, _n):
    
    # construct the empty output image
    output_image = np.empty([cfg.number_of_rows, cfg.number_of_columns])

    # load the trained model
    with open(_model_path, 'r') as f:
        json_model = json.load(f)
        model = model_from_json(json_model)
        model.load_weights(_model_path.replace('.json', '.h5'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    # load scene and its `n` first pixel value data
    scene_path = os.path.join(cfg.folder_scenes_path, _scene_name)

    for id_column in range(cfg.number_of_columns):

        folder_path = os.path.join(scene_path, str(id_column))

        pixels_predicted = []

        for id_row in range(cfg.number_of_rows):
            
            pixel_filename = _scene_name + '_' + str(id_column) + '_' + str(id_row) + ".dat"
            pixel_file_path = os.path.join(folder_path, pixel_filename)
            
            with open(pixel_file_path, 'r') as f:

                # predict the expected pixel value
                lines = [float(l)/255. for l in f.readlines()]
                pixel_values = lines[0:int(_n)]
                pixel_values = np.array(pixel_values).reshape(1, (int(_n)))
                # predict pixel per pixel
                pixels_predicted.append(model.predict(pixel_values))
                
        # change normalized predicted value to pixel value
        pixels_predicted = [ val * 255. for val in pixels_predicted]

        for id_pixel, pixel in enumerate(pixels_predicted):
            output_image[id_pixel, id_column] = pixel

        print("{0:.2f}%".format(id_column / cfg.number_of_columns * 100))
        sys.stdout.write("\033[F")

    return output_image

def main():

    parser = argparse.ArgumentParser(description="Train model and saved it")

    parser.add_argument('--scene', type=str, help='Scene name to reconstruct', choices=cfg.scenes_list)
    parser.add_argument('--model_path', type=str, help='Json model file path')
    parser.add_argument('--n', type=str, help='Number of pixel values approximated to keep')
    parser.add_argument('--image_name', type=str, help="The ouput image name")

    args = parser.parse_args()

    param_scene_name = args.scene
    param_n = args.n
    param_model_path = args.model_path
    param_image_name = args.image_name

    # get default value of `n` param
    if not param_n:
        param_n = param_model_path.split('_')[0]

    output_image = reconstruct(param_scene_name, param_model_path, param_n)

    if not os.path.exists(cfg.reconstructed_folder):
        os.makedirs(cfg.reconstructed_folder)

    image_path = os.path.join(cfg.reconstructed_folder, param_image_name)

    img = Image.fromarray(np.uint8(output_image))
    img.save(image_path)

if __name__== "__main__":
    main()
