import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, \
    PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from joblib import dump

df_base = pd.read_csv('../dataset/results2/df_base_new.csv')

features = df_base[df_base.columns[3:16].values]
target = df_base[['target']]
target = np.ravel(target)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)

# Random Forest Model
rf_params = {"max_depth": [20, 30], "min_samples_split": [2, 10], "max_leaf_nodes": [100, 175, 200],
             "min_samples_leaf": [2, 5], "n_estimators": [100, 250], "max_features": ["sqrt"]}
rf_model = RandomForestClassifier()
rf_cv = GridSearchCV(rf_model, rf_params)
rf_cv.fit(X_train.values, y_train)
rf_model = rf_cv.best_estimator_
rf_pred_prob1 = rf_model.predict_proba(X_test.values)[:, 1]
rf_pred_prob2 = rf_model.predict_proba(X_train.values)[:, 1]

# Gradient descent model
sgd_params = {"learning_rate": [0.01, 0.1], "min_samples_split": [5, 10], "min_samples_leaf": [3, 5],
              "max_depth": [3, 5, 10], "max_features": ["sqrt"], "n_estimators": [100, 200]}
sgd_model = GradientBoostingClassifier()
sgd_cv = GridSearchCV(sgd_model, sgd_params)
sgd_cv.fit(X_train.values, y_train)
sgd_model = sgd_cv.best_estimator_
sgd_pred_prob1 = sgd_model.predict_proba(X_test.values)[:, 1]
sgd_pred_prob2 = sgd_model.predict_proba(X_train.values)[:, 1]

# Neural network models
mlp_params = {"learning_rate_init": [0.001, 0.01, 0.1], "hidden_layer_sizes": [(100,), (100, 50), (50, 30)],
              "activation": ["relu", "tanh"]}
mlp_model = MLPClassifier(max_iter=500)
mlp_cv = GridSearchCV(mlp_model, mlp_params)
mlp_cv.fit(X_train.values, y_train)
mlp_model = mlp_cv.best_estimator_
mlp_pred_prob1 = mlp_model.predict_proba(X_test.values)[:, 1]
mlp_pred_prob2 = mlp_model.predict_proba(X_train.values)[:, 1]

# Calculating AUC
rf_auc1 = roc_auc_score(y_test, rf_pred_prob1)
sgd_auc1 = roc_auc_score(y_test, sgd_pred_prob1)
mlp_auc1 = roc_auc_score(y_test, mlp_pred_prob1)

rf_auc2 = roc_auc_score(y_train, rf_pred_prob2)
sgd_auc2 = roc_auc_score(y_train, sgd_pred_prob2)
mlp_auc2 = roc_auc_score(y_train, mlp_pred_prob2)

# Mapping the ROC
fpr1, tpr1, _ = roc_curve(y_test, rf_pred_prob1)
f
fpr2, tpr2, _ = roc_curve(y_test, sgd_pred_prob1)
fpr3, tpr3, _ = roc_curve(y_test, mlp_pred_prob1)

fpr1_train, tpr1_train, _ = roc_curve(y_train, rf_pred_prob2)
fpr2_train, tpr2_train, _ = roc_curve(y_train, sgd_pred_prob2)
fpr3_train, tpr3_train, _ = roc_curve(y_train, mlp_pred_prob2)

plt.plot(fpr1, tpr1, label='RandomForest (AUC = {:.2f})'.format(rf_auc1))
plt.plot(fpr2, tpr2, label='GradientBoosting (AUC = {:.2f})'.format(sgd_auc1))
plt.plot(fpr3, tpr3, label='MLP (AUC = {:.2f})'.format(mlp_auc1))
plt.plot(fpr1_train, tpr1_train, label='RandomForest train (AUC = {:.2f})'.format(rf_auc2))
plt.plot(fpr2_train, tpr2_train, label='GradientBoosting train (AUC = {:.2f})'.format(sgd_auc2))
plt.plot(fpr3_train, tpr3_train, label='MLP train (AUC = {:.2f})'.format(mlp_auc2))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('../images/ROC_full.png')
plt.show()

# plt.figure(figsize=(20, 15))
# rf_cm = confusion_matrix(y_test, rf_model.predict(X_test.values))
# plt.imshow(rf_cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()

models = {'Random Forest': rf_model, 'Gradient Boosting': sgd_model, 'MLP': mlp_model}

for name, model in models.items():
    y_pred = model.predict(X_test.values)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name}')
    plt.savefig(f'../images/confusion_matrix_{name}.png')
    plt.show()


def plot_feature_importance(model_name, feature_names, name):
    importance_f = model_name.feature_importances_
    indices = np.argsort(importance_f)
    plt.figure(figsize=(20, 15))
    plt.barh(range(len(indices)), importance_f[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.savefig(f'../images/feature_importance_{name}.png')
    plt.show()


plot_feature_importance(rf_model, features.columns, 'Random Forest')
plot_feature_importance(sgd_model, features.columns, 'Gradient Boosting')

for name, model in models.items():
    y_score = model.predict_proba(X_test.values)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title(f'Precision-Recall Curve for {name}')
    plt.savefig(f'../images/precision_recall_{name}.png')
    plt.show()


def plot_learning_curve(model_name, X, y, name):
    train_sizes, train_scores, test_scores = learning_curve(model_name, X, y, cv=5)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.savefig(f'../images/learning_curve_{name}.png')
    plt.show()


for name, model in models.items():
    plot_learning_curve(model, features.values, target, name)

# save models
dump(rf_model, '../models/rf_model.joblib')
dump(sgd_model, '../models/sgd_model.joblib')
dump(mlp_model, '../models/mlp_model.joblib')
