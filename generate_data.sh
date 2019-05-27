for n in {3,4,5,6,7,8,9,10,15,20,25,30}; do
    for row in {1,2,3,4,5}; do
        for column in {1,2,3,4,5}; do

            # Run creation of dataset and train model
            DATASET_NAME="data/dataset_${n}_column_${column}_row_${row}.csv"

            python make_dataset.py --n ${n} --each_row ${row} --each_column ${column} &
        done
    done
done