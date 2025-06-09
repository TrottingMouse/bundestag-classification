import pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time


RANDOM_STATE = 42


def train_pipeline(model):
    """Input: a scikit-learn machine learning class.
    Returns a pipeline with a scaler and the class."""
    return make_pipeline(preprocessing.StandardScaler(), model)


def train(model, X, y):
    """Inputs: a scikit-learn machine learning class, data and labels.
    Returns a classifier fitted to data and labels."""
    pipe = train_pipeline(model)
    pipe.fit(X, y)
    return pipe


def evaluate(algorithm_name, grids, X_test, y_test):
    """Input: the algorithm name, a dictionary of GridSearchCV objects, test data and labels
    Prints the classification report, the best hyperparameters.
    Saves a confusion matrix for the best model"""
    for dataset in grids:
        grid = grids[dataset]
        y_pred = grid.best_estimator_.predict(X_test)
        print(f"report for {algorithm_name} with training set {dataset}:")
        print(classification_report(y_test, y_pred, digits=3))
        print("Best parameters:", grids[dataset].best_params_)
        print()

    labels = ["CDU/CSU", "SPD", "GRÜNE", "LINKE", "AfD", "FDP"]
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp_cm.plot(cmap='Blues')
    plt.title(f"Confusion Matrix for {algorithm_name}")
    plt.savefig(f"confusion_matrix_{algorithm_name}.png")
    plt.close()




with open('feature_matrices.pkl', 'rb') as infile:
    feature_df = pickle.load(infile)

# remove very small groups (further explanation in written report)
parties_to_remove = ["BSW", "Die Linke", "Fraktionslos"]
filtered_feature_df = feature_df[~feature_df["party"].isin(parties_to_remove)]
X = filtered_feature_df.drop(columns=["party"])
y = filtered_feature_df["party"]


# Perform train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


# Investigate the features

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Set the class_weight parameter in order to not favor the majority class
full_model = train_pipeline(LogisticRegression(class_weight='balanced'))
full_score = cross_val_score(full_model, X, y, cv=cv, scoring='f1_macro').mean()
print(f"macro F1-score with all features: {full_score}")
print()

# check for every feature if the classifier is better without it
for col in X.columns:
    X_mod = X.drop(columns=[col])
    model_mod = train_pipeline(LogisticRegression(class_weight='balanced'))
    score_mod = cross_val_score(model_mod, X_mod, y, cv=cv, scoring='f1_macro').mean()
    print(f"macro F1-score without feature {col}: {score_mod}")
    if score_mod > full_score:
        print(f"Better without feature {col}: {score_mod:.4f} > {full_score:.4f}")


# Build balanced training sets
# Apply undersampling
rus = RandomUnderSampler(random_state=RANDOM_STATE)
X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train, y_train)

# Apply oversampling
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train, y_train)


train_datasets = {"unbalanced": [X_train, y_train],
                  "undersampled": [X_train_undersampled, y_train_undersampled],
                  "oversampled": [X_train_oversampled, y_train_oversampled]}


# Perform tuning of hyperparameters with a grid search
now = time.time()
# Logistic Regression

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'solver': ['liblinear']
}

grids_lr = {}
for dataset in train_datasets.keys():
    grids_lr[dataset] = GridSearchCV(
        # without setting the maximum iterations, there is not convergence
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid_lr,
        scoring='f1_macro',
        cv=5
    )

    grids_lr[dataset].fit(train_datasets[dataset][0], train_datasets[dataset][1])
    print(time.time() - now)


# Support Vector Classifier

param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced', None]
}

grids_svc = {}
for dataset in train_datasets.keys():
    grids_svc[dataset] = GridSearchCV(
        SVC(random_state=42),
        param_grid_svc,
        scoring='f1_macro',
        cv=5
    )

    grids_svc[dataset].fit(train_datasets[dataset][0], train_datasets[dataset][1])
    print(time.time() - now)



# Random Forest Classifier

param_grid_rfc = {
    'n_estimators': [100, 300],
    'max_depth': [10],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [2, 4],
    'class_weight': [None, 'balanced']
}

grids_rfc = {}
for dataset in train_datasets.keys():
    grids_rfc[dataset] = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rfc,
        scoring='f1_macro',
        cv=5
    )

    grids_rfc[dataset].fit(train_datasets[dataset][0], train_datasets[dataset][1])
    print(time.time() - now)


# Evaluate all three algorithms
evaluate("logistic_regression", grids_lr, X_test, y_test)
evaluate("svc", grids_svc, X_test, y_test)
evaluate("random_forest", grids_rfc, X_test, y_test)
