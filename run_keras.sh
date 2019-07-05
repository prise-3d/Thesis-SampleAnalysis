# erase "results/models_comparisons.csv" file and write new header
file_path='results/models_comparisons_keras.csv'

erased=$1

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p models_info
    touch ${file_path}

    # add of header
    echo 'model_name; number_of_approximations; coeff_of_determination; MSE 10 samples; MSE 1000 samples;' >> ${file_path}
fi

for feature in {'variances','samples'}; do
    for n in {3,4,5,6,7,8,9,10,15,20,25,30}; do
    for row in {1,2,3,4,5}; do
        for column in {1,2,3,4,5}; do

                # Run creation of dataset and train model
                DATASET_NAME="data/dataset_${n}_${feature}_column_${column}_row_${row}.csv"
                MODEL_NAME="${n}_${feature}_column_${column}_row_${row}_${model}"
                IMAGE_RECONSTRUCTED="Sponza1_${feature}_${n}_${row}_${column}.png"

                if ! grep -q "${MODEL_NAME}" "${file_path}"; then
                    echo "Run computation for model ${MODEL_NAME}"

                    # Already computed..
                    python make_dataset.py --n ${n} --feature ${feature} --each_row ${row} --each_column ${column}
                    python train_model_keras.py --data ${DATASET_NAME} --model_name ${MODEL_NAME}

                    # TODO : Add of reconstruct process for image ?
                    python reconstruct_keras.py --n ${n} --feature ${feature} --model_path saved_models/${MODEL_NAME}.json --scene Sponza1 --image_name ${IMAGE_RECONSTRUCTED}
                    python write_result_keras.py --n ${n} --feature ${feature} --model_path saved_models/${MODEL_NAME}.json --scene Sponza1 --image_path reconstructed/${IMAGE_RECONSTRUCTED} --data ${DATASET_NAME} --iqa mse &
                else
                    echo "${MODEL_NAME} results already computed.."
                fi
            done
        done
    done
done
