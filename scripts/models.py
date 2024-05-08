from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_curve, roc_auc_score, average_precision_score, make_scorer)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
import pickle
import os
import json
import tkinter.filedialog as fd

class ModelsHandler:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.state = None
        self.models = {}

    def _initialize_models(self):
        self.models = {
            'Random Forest': RandomForestClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'XGBoost': XGBClassifier(verbosity=0)
        }

    def _initialize_meta_classifiers(self):
        voting_clf = VotingClassifier(estimators=[(name, model) for name, model in self.models.items()], voting='soft')
        self.models['Voting'] = voting_clf

    def generate_models(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self._initialize_models()
        self._initialize_meta_classifiers()
        for _, model in self.models.items():
            model.fit(self.X_train, self.y_train)
        
        self.state = "Trained models are\n\n" + str(self.models.keys()) + "\n\n"
        return self.state

    def _load_hyperparameters(self, json_path):
        with open(json_path) as f:
            return json.load(f)

    def hyperparameter_tuning(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self._initialize_models()
        scorers = {
            'precision_score': make_scorer(average_precision_score),
            'accuracy_score': make_scorer(accuracy_score),
            'roc_auc': make_scorer(roc_auc_score)
        }

        json_path = fd.askopenfilename(filetypes=[('json', '.json')], title='Select parameters JSON')
        param_grids = self._load_hyperparameters(json_path)

        text = ""
        for name, model in self.models.items():
            gs = GridSearchCV(model, param_grids[name], n_jobs=-1, cv=KFold(n_splits=5, shuffle=True, random_state=24),
                              scoring=scorers, refit='precision_score', return_train_score=True)
            gs.fit(self.X_train, self.y_train)
            self.models[name] = gs.best_estimator_
            text += f"{gs.best_params_}{gs.best_score_}\n\n"
        print(text)
        self._initialize_meta_classifier()

    def predict(self, X_test):
        self.X_test = X_test
        self.predictions = {}
        for name, model in self.models.items():
            self.predictions[name] = model.predict(X_test)
        return self.predictions

    def validation(self, y_test):
        self.y_test = y_test
        confusion_matrices = {}
        accuracy_scores = {}
        roc_curves = {}
        roc_auc_scores = {}
        self.state = ""
        for name, model in self.models.items():
            confusion_matrices[name] = confusion_matrix(y_test, self.predictions[name], labels=[0, 1])
            self.state += f"{confusion_matrix(y_test, self.predictions[name], labels=[0, 1])}\n\n"

            accuracy_scores[name] = accuracy_score(y_test, self.predictions[name]) * 100
            self.state += f"{name} accuracy: {accuracy_score(y_test, self.predictions[name]) * 100}\n-------------------------------\n\n"

            y_score = model.predict_proba(self.X_test)[:, 1]
            roc_curves[name] = roc_curve(y_test, y_score, pos_label=1, drop_intermediate=False)

            roc_auc_scores[name] = roc_auc_score(y_test, self.predictions[name])

        self.validation_scores = {'confusion_matrix': confusion_matrices, 'accuracy_score': accuracy_scores,
                                  'roc_curve': roc_curves, 'roc_auc': roc_auc_scores}
        return self.state

    def load_models(self):
        self.models = {}
        files = fd.askopenfilenames(filetypes=[('pickle', '.pkl')], title='Select models to load')
        for path in files:
            model = pickle.load(open(path, 'rb'))
            self.models[os.path.basename(path).rstrip(".pkl")] = model
        self.state = f"Models {str(self.models.keys())} have been loaded\n\n"
        return self.state

    def save_models(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        for name, model in self.models.items():
            path = os.path.join("models", name + ".pkl")
            pickle.dump(model, open(path, 'wb'))
        self.state = f"Models {str(self.models.keys())} saved at \models.\n\n"
        return self.state
