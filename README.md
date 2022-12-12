# Neural_Network_Charity_Analysis

## Project Overview
A Neural Network is built with TensorFlow and Keras to predict the of outcome of funding a particular charitable investment.
Applicant data is take from AlphabetSoup for 34,200 funding campaigns. The data is processed, encoded, and scaled, then used to train a sequential model. The model is then evaluated. Several rounds model tuning are performed to attempt to optimize the model. A goal of 75% model accuracy has been selected for the final model.


### Resources
- Data Source: Resources\charity_data.csv
- Software: Jupyter Notebook, python 3.7.15, TensorFlow 2.11.0, Keras 2.11.0, Sklearn 1.0.1, pandas 1.3.5, shap 0.41.0

---

## Results
The data is read from csv into a pandas dataframe for preprocessing:
AvailabLe variables: EIN, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, IS_SUCCESSFUL. 
    
1. Data is segmented into target and feature variables, and unecessary columns are dropped.   
    - Unnecessary columns EIN and NAME removed.
    - IS_SUCCESSFUL is selected as the target variable. The data is relatively balanced around the target variable.
    - Features are: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT


4. Binning is used to reduce unique values in APPLICATON_TYPE and CLASSIFICATION columns
    - There are 17 unique application types, 27,037 applications used the T3 application type. Rare occurences are joined into an 'other' category to reduce unique values in application type to 9. Application types with less than 500 appearances are grouped together. 
    - There are 71 unique Classification values: 17,326 campaings have the C1000 classification. Rare occurences are joined into an 'other' category to reduce unique values to 6. All classifications with less than 1500 appearances are grouped together.


5. Categorical values are encoded using OneHotEncoder.


6. The data is split into training and testing sets using train_test_split, and scaled using StandardScaler.
    - After encoding, splitting, and scaling the data has 43 independent variables, and 1 target variable.
    - 25,724 of 35,200 campaigns are in the training set, the remainder are used for testing.


7. The primary model is constructed using tf.keras.Sequential with:
    - an input layer with 43 features
    - 2 hidden layers, 1 output layer
    - layer 1 contains 80 neurons, and the ReLU activation function.
    - layer 2 contains 30 neurons, and the ReLU activation function
    - Output layer has 1 node and the sigmoid activation function.


8. The primary model is compiled and trained for 100 epochs using checkpoints to save weights every 5 epochs. Weights are saved under checkpoints/Original_model. The model is saved under models/Original_model.


The primary model is evaluated:
![primary_model_score](https://github.com/Jforbus/Neural_Network_Charity_Analysis/blob/main/Resources/primary_model_score.png)
### This model does not meet the accuracy goal.

---

9. Optimization Trials: The original model will be tested with each of the following changes made independently. Weights and Model instances will be saved for each.

- SHAP is used to calculate and visualize feature importance during optimization to guide data reprocessing.

    ![shap_code](https://github.com/Jforbus/Neural_Network_Charity_Analysis/blob/main/Resources/shap_value_code.png)
    ![shap_bar_values](https://github.com/Jforbus/Neural_Network_Charity_Analysis/blob/main/Resources/shap_value_bar.png)


    1. Add 3rd hidden layer with 15 nodes
        - The addition of a third layer does not add value to the model, and did not increase accuracy.
    
    
    2. Decrease neurons in original model.
        - Decreasing the neurons did hindered the model slightly
    
    
    3. Change activation function to Tanh.
        - The tanh activation function matches the scaling of StandardScaler better, but did not improve model performance.
    
    
    4. Increase number of bins for APPLICATION_TYPE.
        - cutoff for 'other' category is reduced to 100 appearances. This change does not improve model performance.


    5. Drop APPLICATION_TYPE.
        - Dropping Application type marginally hinders the model.


    6 . Additional Neurons per Layer, Additional Layer, Filter for most important Variables, Decrease bins for Class and App type.
        - These changes do not benefit model performance.

    7. Increase number of bins for CLASSIFICATION.
        - These changes do not benefit model performance.


    8. Change scaler to better match ReLU activation function, keep and bin NAME column, bin ASK_AMT, increase neurons, add third layer.
        - The name column under further review has a fair amount of repeated instances, and is kept in this training set.
        - The ask_amt column is binned into 5 categories.
        - The scaler is changed to MinMaxScaler, better matching the ReLU activation function by removing negative values.
        - Nuerons are increased to match suggested amount of 2X features in the first layer. followed by 1X and .5X.
        - This model acheives the goal, reaching 77.3% accuracy with 59% loss.

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
![trial_7_score](https://github.com/Jforbus/Neural_Network_Charity_Analysis/blob/main/Resources/trial_7_score.png)


### Trial 8 is successful in increasing accuracy above 75%, as well as decreasing loss.
Model is saved under models/optimized_model, weights under checkpoints/trial_8
![Final_model_evaluation](https://github.com/Jforbus/Neural_Network_Charity_Analysis/blob/main/Resources/Final_model_evaluation.png)

---

## Summary
The original model had decent performance but failed to meet the goal of 75% accuracy. The successful model implemented the following changes: MinMaxScaler is used to scale data within range of 0-1 to better match relu activation function, the name column is kept and has binning performed, the ask_amt column has binning applied and, Neurons are increased significantly, and an additional layer is added. This final model achieved 77.3% accuracy with 59% loss.

Further testing reveals the addition of the name column had the most to do with the increased accuracy, and may be evidence that we are simply overfitting the present data. Overall the Neural Network had decent performance predicting the campaign outcomes, but leaves a lot to be desired. Further adjustments to the model may help sufficiently match the complexity of the data, or preprocessing changes may lead to better model performance. 

Another potential is to try a different type of model. A decision tree classifier, such as RandomForest, may have more success in this situtation. RandomForestClassifier will process the data much faster, has a more interpretable process that will aid in optimization, and is resistant to overfitting.
