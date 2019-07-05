# erase "results/models_comparisons.csv" file and write new header
file_path='results/models_comparisons.csv'

erased=$1

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p models_info
    touch ${file_path}

    # add of header
    echo 'model_name; number_of_approximations; coeff_of_determination;' >> ${file_path}
fi

for feature in {'variances','samples'}; do
    for n in {3,4,5,6,7,8,9,10,15,20,25,30}; do
        for row in {1,2,3,4,5}; do
            for column in {1,2,3,4,5}; do

                # Run creation of dataset and train model
                DATASET_NAME="data/dataset_${n}_${feature}_column_${column}_row_${row}.csv"

                if ! grep -q "${MODEL_NAME}" "${file_path}"; then
                    echo "Run computation data for model ${MODEL_NAME}"

                    python make_dataset.py --n ${n} --feature ${feature} --each_row ${row} --each_column ${column}
                fi

                for model in {"SGD","Ridge"}; do

                    MODEL_NAME="${n}_${feature}_column_${column}_row_${row}_${model}"

                    if ! grep -q "${MODEL_NAME}" "${file_path}"; then
                        echo "Run computation for model ${MODEL_NAME}"

                        python train_model.py --data ${DATASET_NAME} --model ${model}
                    else
                        echo "${MODEL_NAME} results already computed.."
                    fi
                done
            done
        done
    done
done
