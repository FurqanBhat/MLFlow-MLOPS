import  mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub



dagshub.init(repo_owner='FurqanBhat', repo_name='MLFlow-MLOPS', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/FurqanBhat/MLFlow-MLOPS.mlflow')

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.10, random_state=42)

max_depth = 15
n_estimators = 10

mlflow.set_experiment('EXP 1')

with mlflow.start_run():
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy',  accuracy)
    mlflow.log_params({'max_depth': max_depth, 'n_estimators': n_estimators})

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    plt.savefig('cf_matrix.png')

    mlflow.log_artifact('cf_matrix.png')
    mlflow.log_artifact(__file__)  #__file__ is used to mention current file

    mlflow.set_tags({'author': 'furqan', 'project': ' wine classificaiton'})

    mlflow.sklearn.log_model(rf, 'random-forest-model')


    print(accuracy)
