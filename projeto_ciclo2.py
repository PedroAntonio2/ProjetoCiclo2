import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from deslib.dcs import OLA, LCA, MCB
from deslib.des import KNORAE, KNORAU


# 1) Load the dataset

columns_to_drop = ['age', 'Unnamed: 0.1', 'Unnamed: 0', 'split', 'signalName', 'classe']

folds = 10
kf = StratifiedKFold(n_splits = folds)

scaler = StandardScaler()


class NormalModelCreator():
    def __init__(self, model_type, feature_selection = False, **params):
        self.model_type = model_type
        self.params = params
        self.feature_selection = feature_selection
        self.model = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.auc_score = None
        self.specificity = None
        self.run_id = None

    def fit_and_log(self):
        if self.model_type == 'MLP':
            self.model = MLPClassifier(random_state=123, **self.params)
        elif self.model_type == 'DecisionTree':
            self.model = DecisionTreeClassifier(random_state=123, **self.params)
        elif self.model_type == 'KNN':
            self.model = KNeighborsClassifier(**self.params)
        elif self.model_type == 'RandomForest':
            self.model = RandomForestClassifier(random_state=123, **self.params)
        elif self.model_type == 'Bagging':
            self.model = BaggingClassifier(random_state=123, **self.params)
        elif self.model_type == 'GradientBoosting':
            self.model = GradientBoostingClassifier(random_state=123, **self.params)
        elif self.model_type == 'XGBoost':
            self.model = XGBClassifier(random_state=123, **self.params)
        elif self.model_type == 'LightGBM':
            self.model = LGBMClassifier(random_state=123, **self.params)
        else:
            print("Invalid model type")

        with mlflow.start_run(run_name=f"{self.model_type} Run"):
            results_accuracy = []
            results_precision = []
            results_recall = []
            results_auc_score = []
            results_specificity = []
            
            for i in range(folds):
                self.model.fit(X_train[i], y_train[i])
                y_pred = self.model.predict(X_test[i])

                tn, fp, fn, tp = confusion_matrix(y_test[i], y_pred).ravel()
                specificity = tn / (tn + fp)

                results_accuracy.append(accuracy_score(y_test[i], y_pred))
                results_precision.append(precision_score(y_test[i], y_pred))
                results_recall.append(recall_score(y_test[i], y_pred))
                results_auc_score.append(roc_auc_score(y_test[i], y_pred))
                results_specificity.append(specificity)

            self.accuracy = np.mean(results_accuracy)
            self.precision = np.mean(results_precision)
            self.recall = np.mean(results_recall)
            self.auc_score = np.mean(results_auc_score)
            self.specificity = np.mean(results_specificity)

            #Log metrics
            mlflow.log_metrics({"accuracy": self.accuracy, "precision": self.precision, "recall": self.recall, "auc_score": self.auc_score, "specificity": self.specificity})

            #Log parameters
            mlflow.log_params(self.params)

            mlflow.set_tag("description", f"Model: {self.model_type}, Feature Selection: {self.feature_selection}")

            mlflow.sklearn.log_model(self.model, self.model_type)
            self.run_id = mlflow.active_run().info.run_id

            metrics = {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "auc_score": self.auc_score,
                "specificity": self.specificity,
                "run_id": self.run_id,
                "model_type": self.model_type
            }

            return metrics


class DynamicSelectionModelCreator():
    def __init__(self, model_type, model_base, feature_selection = False, **params):
        self.model_type = model_type
        self.model_base = model_base
        self.feature_selection = feature_selection
        self.params = params
        self.model = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.auc_score = None
        self.specificity = None
        self.run_id = None

    def fit_and_log(self):
        if self.model_type == 'OLA':
            self.model = OLA(self.model_base, **self.params)
        elif self.model_type == 'LCA':
            self.model = LCA(self.model_base, **self.params)
        elif self.model_type == 'KNORAU':
            self.model = KNORAU(self.model_base, **self.params)
        elif self.model_type == 'KNORAE':
            self.model = KNORAE(self.model_base, **self.params)
        elif self.model_type == 'MCB':
            self.model = MCB(self.model_base, **self.params)
        else:
            print("Invalid model type")

        with mlflow.start_run(run_name=f"{self.model_type} Run"):
            results_accuracy = []
            results_precision = []
            results_recall = []
            results_auc_score = []
            results_specificity = []
            
            for i in range(folds):
                X_new, X_dsel, y_new, y_dsel = train_test_split(X_train[i], y_train[i], test_size=0.5,random_state=123)
                self.model.fit(X_dsel, y_dsel)
                y_pred = self.model.predict(X_test[i])

                tn, fp, fn, tp = confusion_matrix(y_test[i], y_pred).ravel()
                specificity = tn / (tn + fp)

                results_accuracy.append(accuracy_score(y_test[i], y_pred))
                results_precision.append(precision_score(y_test[i], y_pred))
                results_recall.append(recall_score(y_test[i], y_pred))
                results_auc_score.append(roc_auc_score(y_test[i], y_pred))
                results_specificity.append(specificity)

            self.accuracy = np.mean(results_accuracy)
            self.precision = np.mean(results_precision)
            self.recall = np.mean(results_recall)
            self.auc_score = np.mean(results_auc_score)
            self.specificity = np.mean(results_specificity)

            #Log metrics
            mlflow.log_metrics({"accuracy": self.accuracy, "precision": self.precision, "recall": self.recall, "auc_score": self.auc_score, "specificity": self.specificity})

            #Log parameters
            mlflow.log_params({"k": 5, "Generalization": self.model_base})

            mlflow.set_tag("description", f"Model: {self.model_type}, Generalization: {self.model_base}, Feature Selection: {self.feature_selection}")

            mlflow.sklearn.log_model(self.model, self.model_type)
            self.run_id = mlflow.active_run().info.run_id

            metrics = {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "auc_score": self.auc_score,
                "specificity": self.specificity,
                "run_id": self.run_id,
                "model_type": self.model_type
            }

            return metrics

mlflow.set_experiment("exp_projeto_ciclo_2_3.1")

metrics = {}
results = []

for i in range(2):
    if i == 0:
        feature_selection = False
        df = pd.read_csv(f'dados_ptb_with_age_sex.csv')

        y = df['classe']
        X = df.drop(columns_to_drop, axis = 1)

        X = scaler.fit_transform(X)

        X_train = []
        y_train = []

        X_test = []
        y_test = []

        for train_index, test_index in kf.split(X,y):
            X_train.append(X[train_index])
            X_test.append(X[test_index])

            y_train.append(y[train_index])
            y_test.append(y[test_index])
    else:
        feature_selection = True
        
        df = pd.read_csv(f'dados_ptb_with_age_sex.csv')

        y = df['classe']
        X = df.drop(columns_to_drop, axis = 1)

        
        # O SelectKBest pode ser usado com várias funções, aqui usamos a f_classif (dados numéricos e variável alvo categórica)
        f_classif = SelectKBest(score_func=f_classif,k=10)

        modeloclassif = f_classif.fit(X,y)
        modeloclassif.get_feature_names_out()

        # identificando os nomes das features escolhidas
        cols = modeloclassif.get_support(indices=True)
        X.iloc[:,cols].columns

        X = scaler.fit_transform(X)

        X_train = []
        y_train = []

        X_test = []
        y_test = []

        for train_index, test_index in kf.split(X,y):
            X_train.append(X[train_index])
            X_test.append(X[test_index])

            y_train.append(y[train_index])
            y_test.append(y[test_index])


    model1 = NormalModelCreator('MLP', feature_selection, hidden_layer_sizes=(100, 100, 2), max_iter=1000)
    metrics = model1.fit_and_log()
    results.append(metrics)

    model2 = NormalModelCreator('MLP', feature_selection, hidden_layer_sizes=(80, 50, 2), max_iter=2500)
    metrics = model2.fit_and_log()
    results.append(metrics)

    model3 = NormalModelCreator('MLP', feature_selection, hidden_layer_sizes=(100, 50, 2), max_iter=5000)
    metrics = model3.fit_and_log()
    results.append(metrics)

    model4 = NormalModelCreator('DecisionTree', feature_selection, criterion='entropy', max_depth=5)
    metrics = model4.fit_and_log()
    results.append(metrics)

    model5 = NormalModelCreator('DecisionTree', feature_selection, criterion='gini', max_depth=5)
    metrics = model5.fit_and_log()
    results.append(metrics)

    model6 = NormalModelCreator('DecisionTree', feature_selection, criterion='entropy', max_depth=10)
    metrics = model6.fit_and_log()
    results.append(metrics)

    model7 = NormalModelCreator('KNN', feature_selection, n_neighbors=5)
    metrics = model7.fit_and_log()
    results.append(metrics)

    model8 = NormalModelCreator('KNN', feature_selection, n_neighbors=10)
    metrics = model8.fit_and_log()
    results.append(metrics)

    model9 = NormalModelCreator('KNN', feature_selection, n_neighbors=15)
    metrics = model9.fit_and_log()
    results.append(metrics)

    model10 = NormalModelCreator('RandomForest', feature_selection)
    metrics = model10.fit_and_log()
    results.append(metrics)

    model11 = NormalModelCreator('Bagging', feature_selection)
    metrics = model11.fit_and_log()
    results.append(metrics)

    model12 = NormalModelCreator('GradientBoosting', feature_selection)
    metrics = model12.fit_and_log()
    results.append(metrics)

    model13 = NormalModelCreator('XGBoost', feature_selection)
    metrics = model13.fit_and_log()
    results.append(metrics)

    model14 = NormalModelCreator('LightGBM', feature_selection)
    metrics = model14.fit_and_log()
    results.append(metrics)

    rf = RandomForestClassifier(random_state=123)
    rf.fit(X_train[9], y_train[9])

    model15 = DynamicSelectionModelCreator('OLA', rf, feature_selection, k=5)
    metrics = model15.fit_and_log()
    results.append(metrics)

    model16 = DynamicSelectionModelCreator('LCA', rf, feature_selection, k=5)
    metrics = model16.fit_and_log()
    results.append(metrics)

    model17 = DynamicSelectionModelCreator('KNORAU', rf, feature_selection, k=5)
    metrics = model17.fit_and_log()
    results.append(metrics)

    model18 = DynamicSelectionModelCreator('KNORAE', rf, feature_selection, k=5)
    metrics = model18.fit_and_log()
    results.append(metrics)

    model19 = DynamicSelectionModelCreator('MCB', rf, feature_selection, k=5)
    metrics = model19.fit_and_log()
    results.append(metrics)

    bagging = BaggingClassifier(random_state=123)
    bagging.fit(X_train[9], y_train[9])

    model20 = DynamicSelectionModelCreator('OLA', bagging, feature_selection, k=5)
    metrics = model20.fit_and_log()
    results.append(metrics)

    model21 = DynamicSelectionModelCreator('LCA', bagging, feature_selection, k=5)
    metrics = model21.fit_and_log()
    results.append(metrics)

    model22 = DynamicSelectionModelCreator('KNORAU', bagging, feature_selection, k=5)
    metrics = model22.fit_and_log()
    results.append(metrics)

    model23 = DynamicSelectionModelCreator('KNORAE', bagging, feature_selection, k=5)
    metrics = model23.fit_and_log()
    results.append(metrics)

    model24 = DynamicSelectionModelCreator('MCB', bagging, feature_selection, k=5)
    metrics = model24.fit_and_log()
    results.append(metrics)

# Find the 3 best models by accuracy
results_df = pd.DataFrame(results)
results_df.sort_values(by='accuracy', ascending=False, inplace=True)

# Register the top 3 models
n_models = 3
for i in range(n_models):
    model_type = results_df.iloc[i]['model_type']
    run_id = results_df.iloc[i]['run_id']

    model_uri = f"runs:/{run_id}/{model_type}"
    model_name = f"{model_type}"

    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Registered model: {model_name} from run {run_id}")