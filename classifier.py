import pickle
import numpy as np
import pandas as pd
import random
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt

def train(model, X, y):
    pipe = make_pipeline(preprocessing.StandardScaler(), model)
    pipe.fit(X, y)
    return pipe


with open('feature_matrices.pkl', 'rb') as infile:
    feature_df = pickle.load(infile)

# remove very small groups (further explanation in written report)
parties_to_remove = ["BSW", "Die Linke", "Fraktionslos"]
filtered_feature_df = feature_df[~feature_df["Party"].isin(parties_to_remove)]
X = filtered_feature_df.drop(columns=["Party"])
y = filtered_feature_df["Party"]


# baseline: 0.26412491377688596
X_train, X_devtest, y_train, y_devtest = train_test_split(X, y, test_size=0.3)
X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=0.33)




# pipe_logrec = train(LogisticRegression(), X_train, y_train)
# all_features_score = pipe_logrec.score(X_test, y_test)
# n_features = len(X_train[0])
# for feature in range(n_features):
#     X_train_modified = [X_train[sample][0:feature]+X_train[sample][feature+1:n_features] for sample in X_train]
#     X_dev_modified = [X_dev[sample][0:feature]+X_dev[sample][feature+1:n_features] for sample in X_dev]
#     pipe_logrec_modified = train(LogisticRegression(), X_train_modified, y_train)
#     modified_features_score = pipe_logrec.score(X_dev, y_dev)
#     if modified_features_score > all_features_score:
#         print(f"The score is better without feature {feature}.")


# building a balanced training set

rus = RandomUnderSampler(random_state=42)

# Apply undersampling
X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)

# Combine features and labels into one balanced DataFrame
#balanced_feature_matrices = pd.concat([X_balanced_df, y_balanced_df], axis=1)

# Display class distribution
print(y_train_balanced.value_counts())
print(y_train.value_counts())






pipe_logrec = train(LogisticRegression(), X_train, y_train)
y_pred = pipe_logrec.predict(X_dev)
print(confusion_matrix(y_dev, y_pred, labels=["CDU/CSU","SPD","BÜNDNIS 90/DIE GRÜNEN","DIE LINKE","AfD","FDP"]))
print(f"Accuracy for Logistic regression (unbalanced dataset): {accuracy_score(y_dev, y_pred)}")
print(f"Balanced accuracy for Logistic regression (unbalanced dataset): {balanced_accuracy_score(y_dev, y_pred)}")


pipe_logrec_balanced = train(LogisticRegression(), X_train_balanced, y_train_balanced)
y_pred_balanced = pipe_logrec_balanced.predict(X_dev)
print(confusion_matrix(y_dev, y_pred_balanced, labels=["CDU/CSU","SPD","BÜNDNIS 90/DIE GRÜNEN","DIE LINKE","AfD","FDP"]))
print(f"Accuracy for Logistic regression (balanced dataset): {accuracy_score(y_dev, y_pred_balanced)}")
print(f"Balanced accuracy for Logistic regression (balanced dataset): {balanced_accuracy_score(y_dev, y_pred_balanced)}")


pipe_svm = train(SVC(), X_train, y_train)
y_pred_svm = pipe_svm.predict(X_dev)
print(f"Accuracy for SVM (unbalanced dataset): {accuracy_score(y_dev, y_pred_svm)}")
print(f"Balanced accuracy for SVM (unbalanced dataset): {balanced_accuracy_score(y_dev, y_pred_svm)}")


pipe_svm_balanced = train(SVC(), X_train_balanced, y_train_balanced)
y_pred_svm_balanced = pipe_svm_balanced.predict(X_dev)
print(f"Accuracy for SVM (balanced dataset): {accuracy_score(y_dev, y_pred_svm_balanced)}")
print(f"Balanced accuracy for SVM (balanced dataset): {balanced_accuracy_score(y_dev, y_pred_svm_balanced)}")



pipe_rfc = train(RandomForestClassifier(), X_train, y_train)
print(pipe_rfc.score(X_test, y_test))

#balanced score?



