# erase "models_info/models_comparisons.csv" file and write new header
file_path='models_info/models_comparisons.csv'

erased=$1

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p models_info
    touch ${file_path}

    # add of header
    echo 'model_name; coeff_of_determination;' >> ${file_path}
fi

for model in {"SGD","Ridge","SVR"}; do
    for row in {7,8,9,10}; do
        for column in {7,8,9,10}; do

            # Run creation of dataset and train model
            DATASET_NAME="data/dataset_10_column_${column}_row_${row}.csv"

            python make_dataset.py --n 10 --each_row ${row} --each_column ${column}
            python train_model.py --data ${DATASET_NAME} --model ${model}
        done
    done
done