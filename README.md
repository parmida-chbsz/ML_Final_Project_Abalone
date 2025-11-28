# Abalone Sex Classification Using Machine Learning
Student: Parmida Choubsaz


## Abstract

This project uses the UCI Abalone data set to predict the sex of abalone (M, F, I) using machine learning. After preprocessing, feature selection and correlation filtering, the final dataset had seven numeric features and one categorical feature. The PyCaret automated classification workflow was applied to train and compare various models. Logistic Regression provided the best performance in cross-validation with an accuracy of 0.555, AUC 0.755, balanced performance in terms of precision, recall and F1-score. Although it is known that the data set has significant class overlap, the model shows reasonable performance for a difficult biological classification task. All code, visualizations, saved models and predictions are included in the accompanying Jupyter Notebook and GitHub repository.

## Literature Review

Paper 1 — Waugh (1995)
Waugh (1995) examined a number of models for classifying abalone and reported that the data set is highly overlapping and classification is inherently difficult. Using the same features that are available in UCI data set, Cascade-Correlation yielded 26.25% accuracy, C4.5 yielded 21.5% accuracy and KNN(k = 5) yielded 3.57% accuracy. These very low values emphasizes the fact that traditional ML models have a hard time differentiating between the three sex classes due to the biological similarity between male, female and infant abalone. Our model's performance (~55%) is better than these early benchmarks and is consistent with more modern results.

Paper 2 — Clark, Schreter & Adams (1996)
Clark et al. (1996) compared neural networks and the Dystal algorithm for the same abalone data. Below are the accuracy results of the two methods, backpropagation and Dystal, on the age groups: they achieved 64% and 55% accuracy, respectively. Their study highlights the challenge of the data set and the need for careful preprocessing. Their reported accuracy of 55-64% for related classification tasks is similar to the performance obtained in this project and confirms that mid-range accuracies are to be expected.


Paper 3 — Barrera-Hernandez et al. (2021)
Barrera-Hernandez et al. (2021) conducted feature-selection and correlation analysis on the abalone dataset and showed that removing features of low importance (such as Rings and Height) helps to increase model performance. After removing two features that did not show high correlation and removing outliers, their optimized dataset helped to improve the accuracy of their classification based on the models. This work informed the method of feature selection in this project where a correlation matrix was employed to remove low contributing attributes prior to comparing models.

## Feature Engineering

All eight original features were discussed, one categorical feature (Sex), and seven numerical shell and weight measurements. A correlation matrix was calculated to determine which features were low contributing and Height and Rings were excluded because of low correlation and low predictive value reported in published literature. The categorical variable sex was label encoded by PyCaret automatically. There were no missing values in the dataset, however numerical features were normalized with z-score scaling during the setup of PyCaret. The final dataset included seven predictive features and 4057 samples after applying feature filtering.

## Performance Metric Selection

Accuracy was chosen as the main performance measure because the target variable (Sex: M, F, I) is a balanced multi-class problem and the accuracy gives an easy to interpret measure of total number of correct predictions. Additionally, the objective of the task is classification rather than ranking or probability calibration, so accuracy is appropriate. Secondary metrics such as AUC, F1 and recall were also reviewed to understand model behaviour across classes, but accuracy was still the most appropriate metric for selecting the final model.

## Experimental Results

PyCaret's compare_models() function evaluated multiple algorithms using 10-fold stratified cross-validation. Logistic Regression gave the best overall performance with an accuracy of 55.5% and area under the curve Area Under the Curve(AUC) of 0.755 and balanced the rate of precision/recall between all classes. Other models like KNN, SVM, Naive Bayes and tree-based were less stable and less generalized. Logistic Regression was therefore selected as the final model and saved as abalone_sex_classifier.pkl.

## Conclusions

This project proves that it is inherently difficult to classify the sex of abalone by physical assessments because the biological characteristics overlap. Despite this, the final model was able to achieve performance in line with previously published results. The project also demonstrated the usefulness of feature selection and correlation filtering for better performance. PyCaret was a very efficient library for quick prototyping and model comparison, however, other libraries like scikit-learn, XGBoost or LightGBM could be explored for more advanced custom modelling in future work.

## Link to video presentation
https://www.loom.com/share/dd90d95db6954f8eaa95a7b07e669703
