# Sample Analysis

## Description

The aim of this project is to predict the mean pixel value from monte carlo process rendering in synthesis images using only few samples information in input for model.


### Data

Data are all scenes samples information obtained during the rendering process.

For each pixel we have a list of all grey value estimated (samples).

### Models
List of models tested :
- Ridge Regression
- SGD
- SVR (with rbf kernel)


## How to use

First you need to contact **jerome.buisine@univ-littoral.fr** in order to get datatset version. The dataset is not available with this source code.


```bash
python make_dataset.py --n 10 --each_row 8 --each_column 8
```

```bash
python reconstruct.py --scene Scene1 --model_path saved_models/Model1.joblib --n 10 --image_name output.png
```

