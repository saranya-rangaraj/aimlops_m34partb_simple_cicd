import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from iris_model.config.core import config
from iris_model.pipeline import iris_pipe
from iris_model.processing.data_manager import load_dataset, save_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import optuna

def run_pipeline_training() -> None: 
    """
    Train the model using config driven static model parameters.
    """
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # random seed for reproducibility
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    iris_pipe.fit(X_train, y_train)
    y_pred = iris_pipe.predict(X_test)
    print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= iris_pipe)
    
def run_pipeline_training_with_hyperparameter_optimization() -> None: 
    """
    Train the model with optuna hyperparameter optimization.
    """
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # random seed for reproducibility
        random_state=config.model_config.random_state,
    )

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 2, 50)
        max_depth = int(trial.suggest_int("max_depth", 2, 10, log=True))
        pca_components = trial.suggest_int("n_components", 3, 4)
    
        iris_pipe = Pipeline([
            ('pca', PCA(n_components=pca_components)),
            ('model_rf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            random_state=config.model_config.random_state))
        ])
        return cross_val_score(
            iris_pipe, X_train, y_train, n_jobs=-1, cv=3
        ).mean()

    # hyperparameter optimization and best study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    trial = study.best_trial
    print("Optuna Best hyperparameters: {}".format(trial.params))

    # Pipeline fitting using best parameters
    iris_best_pipe = Pipeline([
            ('pca', PCA(n_components=trial.params['n_components'])),
            ('model_rf', RandomForestClassifier(n_estimators=trial.params['n_estimators'], max_depth=trial.params['max_depth'],
                                            random_state=config.model_config.random_state))
        ])
    
    iris_best_pipe.fit(X_train, y_train)
    y_pred = iris_best_pipe.predict(X_test)
    print("Test Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= iris_best_pipe)
    
if __name__ == "__main__":
    run_pipeline_training_with_hyperparameter_optimization()