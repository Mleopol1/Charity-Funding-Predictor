# Charity Funding Predictor

This project aims to build a machine learning model that predicts whether or not a charity organization will be successful if funded by investors. The model will be built using the following steps:

1. Data Preprocessing
2. Feature Selection
3. Model Training
4. Model Evaluation

## Data Preprocessing

The dataset used in this project is the "charity_data.csv" which was imported from the "Resources" folder and read into a pandas DataFrame in the "AlphabetSoupCharity.ipynb" Jupyter Notebook. The dataset contains 12 columns, with "IS_SUCCESSFUL" being the target variable, and the remaining columns are features, except "EIN" and "NAME" which were dropped as they were not beneficial to the analysis.

The number of unique values in each column was determined using the `nunique()` function. It was observed that the "APPLICATION_TYPE" and "CLASSIFICATION" columns had too many unique values. Thus, the "APPLICATION_TYPE" column was binned by replacing application types with less than 200 counts with "Other". Similarly, the "CLASSIFICATION" column was binned by replacing classifications with less than 1000 counts with "Other".

The data was then split into training and testing datasets using `train_test_split()` function from sklearn.model_selection. The data was then scaled using StandardScaler from sklearn.preprocessing to standardize the data for better model performance.

## Feature Selection

The features for the model were all columns except the target variable "IS_SUCCESSFUL" and "EIN" and "NAME" columns which were dropped earlier. The model will be built using TensorFlow, which requires numerical input features. Therefore, categorical features were encoded using one-hot encoding using `pd.get_dummies()` function.

## Model Training

The model was built using a neural network with two hidden layers. The first and second hidden layers had 80 and 30 neurons, respectively, with "relu" activation function. The output layer had a single neuron and "sigmoid" activation function.

The model was compiled using "binary_crossentropy" loss function and "adam" optimizer. The model was trained for 100 epochs.

## Model Evaluation

The model was evaluated using the test dataset. The accuracy score and loss value were calculated using the `model.evaluate()` function. The model achieved an accuracy score of around 73.1%.

## Model Improvement

In the "AlphabetSoupCharity_Optimization.ipynb" Jupyter Notebook, an attempt is made to optimize the original machine learning model. A third hidden layer is added with 30 neurons, 20 neurons are added to the second hidden layer, taking it up from 30 neurons to 50 neurons, and 100 epochs were added to the model training. This attempt at improving the model backfired, bringing the accuracy down to around 72.7%.

## Conclusion

This project aimed to predict whether a charity organization will be successful if funded by investors. The machine learning model built using TensorFlow achieved an accuracy score of around 73.1%, and the attempt at optimizing the model achieved an accuracy score of around 72.7%. Further model improvement could be achieved by exploring different neural network architectures and hyperparameters.
