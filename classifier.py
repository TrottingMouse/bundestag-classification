import pickle
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def train(model, X, y):
    pipe = make_pipeline(preprocessing.StandardScaler(), model)
    pipe.fit(X, y)
    return pipe


with open('feature_matrices.pkl', 'rb') as infile:
    feature_matrices = pickle.load(infile)

X = []
y = []


for key in feature_matrices.keys():
    X += feature_matrices[key]
    y_values = [key]*len(feature_matrices[key])
    y += y_values


# baseline: 0.26412491377688596
X_train, X_devtest, y_train, y_devtest = train_test_split(X, y, test_size=0.3)
X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=0.33)


pipe_logrec = train(LogisticRegression(), X_train, y_train)
all_features_score = pipe_logrec.score(X_test, y_test)
n_features = len(X_train[0])
for feature in range(n_features):
    X_train_modified = [X_train[sample][0:feature]+X_train[sample][feature+1:n_features] for sample in X_train]
    X_dev_modified = [X_dev[sample][0:feature]+X_dev[sample][feature+1:n_features] for sample in X_dev]
    pipe_logrec_modified = train(LogisticRegression(), X_train_modified, y_train)
    modified_features_score = pipe_logrec.score(X_dev, y_dev)
    if modified_features_score > all_features_score:
        print(f"The score is better without feature {feature}.")





# building a balanced training set

# Finding out the size of the smallest party in the training set
train_data_sizes = [y_train.count(party) for party in feature_matrices.keys()]
print(train_data_sizes)
min_data_size = min(train_data_sizes)
print(X_train[0])
X_balanced_train = []
y_balanced_train = []
for party in feature_matrices.keys():
    balanced_count = 0
    for i in range(len(X_train)):
        if y_train[i] == party:
            if balanced_count < min_data_size:
                X_balanced_train.append(X_train[i])
                y_balanced_train.append(y_train[i])
            balanced_count += 1

# ensure randomness
balanced_train = list(zip(X_balanced_train, y_balanced_train))
random.shuffle(balanced_train)
X_balanced_train, y_balanced_train = zip(*balanced_train)


pipe_logrec = train(LogisticRegression(), X_train, y_train)
y_pred = pipe_logrec.predict(X_test)
print(confusion_matrix(y_test, y_pred, labels=["CDU/CSU","SPD","BÜNDNIS 90/DIE GRÜNEN","DIE LINKE","AfD","FDP"]))
print(pipe_logrec.score(X_test, y_test))

pipe_logrec_balanced = train(LogisticRegression(), X_balanced_train, y_balanced_train)
y_b_pred = pipe_logrec_balanced.predict(X_test)
print(confusion_matrix(y_test, y_b_pred, labels=["CDU/CSU","SPD","BÜNDNIS 90/DIE GRÜNEN","DIE LINKE","AfD","FDP"]))
print(pipe_logrec_balanced.score(X_test, y_test))

pipe_svm = train(SVC(), X_train, y_train)
print(pipe_svm.score(X_test, y_test))

pipe_svm_balanced = train(SVC(), X_balanced_train, y_balanced_train)
print(pipe_svm_balanced.score(X_test, y_test))


pipe_rfc = train(RandomForestClassifier(), X_train, y_train)
print(pipe_rfc.score(X_test, y_test))

#balanced score?



