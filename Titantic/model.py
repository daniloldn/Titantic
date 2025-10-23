# Script to run model 

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def data_split(df, predictors = ["Sex", "Pclass"], target = 'Survived'):

    VALID_SIZE = 0.2
    train, valid = train_test_split(df, test_size=VALID_SIZE, random_state=42, shuffle=True)

    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values
    return train_X, train_Y, valid_X, valid_Y

def random_forest(train_X, train_Y, valid_X):
    clf = RandomForestClassifier(n_jobs=-1, 
                             random_state=42,
                             criterion="gini",
                             n_estimators=100,
                             verbose=False)
    clf.fit(train_X, train_Y)
    preds_tr = clf.predict(train_X)
    preds = clf.predict(valid_X)

    return preds_tr, preds

def eval(preds_tr, preds, train_y, valid_Y):
    print(metrics.classification_report(train_y, preds_tr, target_names=['Not Survived', 'Survived']))
    print(metrics.classification_report(valid_Y, preds, target_names=['Not Survived', 'Survived']))


def model(df, predictors = ["Sex", "Pclass"], target = 'Survived'):

    train_X, train_Y, valid_X, valid_Y = data_split(df, predictors, target)
    preds_tr, preds = random_forest(train_X, train_Y, valid_X)
    eval(preds_tr, preds, train_Y, valid_Y)
