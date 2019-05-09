# erase "models_info/models_comparisons.csv" file and write new header
file_path='models_info/models_comparisons.csv'

erased=$1

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p models_info
    touch ${file_path}

    # add of header
    echo 'model_name; number_of_approximations; coeff_of_determination;' >> ${file_path}
fi

for n in {10,15,20,25,30}; do
    for model in {"SGD","Ridge","SVR"}; do
        for row in {7,8,9,10}; do
            for column in {7,8,9,10}; do

                # Run creation of dataset and train model
                DATASET_NAME="data/dataset_${n}_column_${column}_row_${row}.csv"
                MODEL_NAME="${n}_column_${column}_row_${row}_${model}"

                if ! grep -q "${MODEL_NAME}" "${file_path}"; then
                    echo "Run computation for model ${MODEL_NAME}"

                    python make_dataset.py --n ${n} --each_row ${row} --each_column ${column}
                    python train_model.py --data ${DATASET_NAME} --model ${model}
                else
                    echo "${MODEL_NAME} results already computed.."
                fi
            done
        done
    done
done
