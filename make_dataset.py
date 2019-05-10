import numpy as np
import pandas as pd

import os, sys, argparse

import modules.config as cfg

def compute_files(_n, _each_row, _each_column):
    """
    Read all folders and files of scenes in order to compute output dataset
    """

    output_dataset_filename = cfg.output_file_prefix + _n + '_column_' + _each_column + '_row_' + _each_row + '.csv'

    output_dataset_filename = os.path.join(cfg.output_data_folder, output_dataset_filename)

    if not os.path.exists(cfg.output_data_folder):
        os.makedirs(cfg.output_data_folder)

    output_file = open(output_dataset_filename, 'w')

    print('Preparing to store data into ', output_dataset_filename)

    scenes = os.listdir(cfg.folder_scenes_path)

    # remove min max file from scenes folder
    scenes = [s for s in scenes if s not in cfg.folder_and_files_filtered]
    scenes = [s for s in scenes if '.csv' not in s] # do not keep generated .csv file

    # skip test scene from dataset
    scenes = [ s for s in scenes if s not in cfg.test_scenes]

    # print(scenes)

    counter = 0
    number_of_elements = len(scenes) * cfg.number_of_rows * cfg.number_of_columns
    #print(number_of_elements, ' to manage')

    for scene in scenes:

        scene_path = os.path.join(cfg.folder_scenes_path, scene)

        for id_column in range(cfg.number_of_columns):

            if id_column % int(_each_column) == 0 :

                folder_path = os.path.join(scene_path, str(id_column))
                
                for id_row in range(cfg.number_of_rows):

                    if id_row % int(_each_row) == 0:

                        pixel_filename = scene + '_' + str(id_column) + '_' + str(id_row) + ".dat"
                        pixel_file_path = os.path.join(folder_path, pixel_filename)

                        saved_row = ''

                        # for each file read content, keep `n` first values and compute mean
                        with open(pixel_file_path, 'r') as f:
                            lines = [float(l)/255. for l in f.readlines()]

                            pixel_values = lines[0:int(_n)]
                            mean = sum(lines) / float(len(lines))

                            saved_row += str(mean)

                            for val in pixel_values:
                                saved_row += ';' + str(val)

                            saved_row += '\n'

                        # store mean and pixel values into .csv row
                        output_file.write(saved_row)

                    counter = counter + 1
            else:
                counter += cfg.number_of_rows

            print("{0:.2f}%".format(counter / number_of_elements * 100))
            sys.stdout.write("\033[F")

    print('\n')
    output_file.close()

def main():

    parser = argparse.ArgumentParser(description="Compute .csv dataset file")

    parser.add_argument('--n', type=str, help='Number of pixel values approximated to keep')
    parser.add_argument('--each_row', type=str, help='Keep only values from specific row', default=1)
    parser.add_argument('--each_column', type=str, help='Keep only values from specific column', default=1)
    args = parser.parse_args()

    param_n = args.n
    param_each_row = args.each_row
    param_each_column = args.each_column

    compute_files(param_n, param_each_row, param_each_column)

if __name__== "__main__":
    main()
