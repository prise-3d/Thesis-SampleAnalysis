from modules.config.global_config import *
import os

# store all variables from global config
context_vars = vars()

# folders
## zone_folder                     = 'zone'
## output_data_folder              = 'data'
## dataset_path                    = 'dataset'
## threshold_map_folder            = 'threshold_map'
## models_information_folder       = 'models_info'
## results_information_folder      = 'results'
## saved_models_folder             = 'saved_models'
## min_max_custom_folder           = 'custom_norm'
## learned_zones_folder            = 'learned_zones'
reconstructed_folder            = 'reconstructed'

# files or extensions
## csv_model_comparisons_filename  = 'models_comparisons.csv'
## seuil_expe_filename             = 'seuilExpe'
## min_max_filename_extension      = '_min_max_values'
output_file_prefix              = "dataset_"
global_result_filepath          = os.path.join(results_information_folder, "models_comparisons.csv")
global_result_filepath_keras    = os.path.join(results_information_folder, "models_comparisons_keras.csv")

# variables 
folder_and_files_filtered       = ["analyse", "make_dataset.py", ".vscode"]

number_of_rows                  = 512
number_of_columns               = 512
keras_epochs                    = 5

kind_of_models                  = ["SGD", "Ridge", "SVR"]


features_list                   = ['samples', 'variances']

scenes_list                     = ['Exterieur01', 'Boulanger', 'CornellBoxNonVide', 'CornellBoxNonVideTextureArcade', 'CornellBoxVide', 'Bar1', 'CornellBoxNonVideTextureDegrade', 'CornellBoxNonVideTextureDamier', 'CornellBoxVideTextureDamier', 'CornellBoxNonVide', 'Sponza1', 'Bureau1_cam2']

test_scenes                     = ['Sponza1']