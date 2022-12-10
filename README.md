# Neural_Network_Charity_Analysis

## Project Overview
A Neural Network is built with TensorFlow and Keras to predict the of outcome of funding a particular charitable investment.
Applicant data is take from AlphabetSoup for 34,000 funding campaigns. The data is process, encoded, and scaled, then used to train a sequential model. The model is then evaluated. Several rounds model tuning will then be performed to attempt to optimize the model. A goal of 75% model accuracy has been selected for the final model.


### Resources
- Data Source: Resources\charity_data.csv
- Software: Jupyter Notebook, python 3.7.15, TensorFlow 2.11.0, Keras 2.11.0, Sklearn 1.0.1, pandas 1.3.5, shap 0.41.0

---

## Results
The data is read from csv into a pandas dataframe for preprocessing:
AvailabLe variables: EIN, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, IS_SUCCESSFUL. 

1. Unnecessary columns are EIN and NAME removed.
2. IS_SUCCESSFUL is selected as the target variable. The data is relatively balanced around the target variable.
3. Features are: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
4. Binning is used to reduce unique values in APPLICATON_TYPE and CLASSIFICATION columns
5. Categorical values are encoded using OneHotEncoder
6. The data is split into training and testing sets using train_test_split, and scaled using StandardScaler
7. The primary model is constructed using tf.keras.Sequential with:
    - 2 hidden layers, 1 output layer
    - layer 1 contains 80 neurons, activation function is ReLU
    - layer 2 contains 30 neurons, activation function is ReLU
    - output layer has 1 node and the sigmoid activation function
8. The primary model is compiled and trained for 100 epochs using checkpoints to save weights every 5 epochs. Weights are saved under checkpoints/Original_model. The model is saved under models/Original_model.
The primary model is evaluated:
![primary_model_score]()
This model does not meet the accuracy goal.

9. Optimization Trials: The original model will be tested with each of the following changes made independently. Weights and Model instances will be saved for each.

    1. Add 3rd hidden layer with 15 nodes
    2. Decrease neurons in original model
    3. Change activation function to Tanh
    4. Shap is used to calculate and visualize feature importance
    ![shap_code]()
    ![shap_bar_values]()
    6. Increase number of bins for APPLICATION_TYPE
    7. Drop APPLICATION_TYPE 
    8. Additional Neurons per Layer, Additional Layer, Filter for most important Variables, Decrease bins for Class and App type.
    9. Increase number of bins for CLASSIFICATION.
    10. Change scaler to better match ReLU activation function, keep and bin NAME column, bin ASK_AMT, increase neurons, add third layer.

    Other indepent tests: several changes applied to trial models to gain insight.
    - ELU, ReLU, LeakyReLU PReLU, ThresholdedReLU, and tanh activation functions, all at 500 epochs with 4 layers - nodes: [150, 70, 30, 15]
    - Adjusted data split to 80% training data
    - MinMaxScaler used to scale data
    - Adam, Nadam, RMSprop, SGD optimizers
    - Layer regulizers: l1, l2, l1_l2, Orthagonal
    - Adjusted learning rate: .03, .05
    - Removal of various features

### Trials 1 - 7 failed to increase accuracy above 73.5%.
Models saved under models/*Trial_x*, weights under checkpoints/*Trial_x*.
![trial_7_score]()
### Trial 8 is successful in increasing accuracy above 75%, as well as decreasing loss.
Model is saved under models/optimized_model, weights under checkpoints/trial_8
![Final_model_evaluation]()

---

## Summary
The original model had decent performance but failed to meet the goal of 75% accuracy. The successful model implemented the following changes: MinMaxScaler is used to scale data within range of 0-1 to better match relu activation function, the name column is kept and has binning performed, the ask_amt column has binning applied and, Neurons are increased significantly, and an additional layer is added. It appears the the addition of the name column had the most to do with the increased accuracy, and may the result of overfitting the present data. Overall the Neural Network had decent performance predicting the campaign outcomes, but leaves a bit to be desired. Further adjustments to the model may help sufficiently match the complexity of the data, or preprocessing changes may lead to a better model. Another potential is to try a different type of model. A RandomForest classifier may have success in this situtation, with the name column and other less influential features removed to decrease the number of features.
