#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator


# # A) Load and preprocess Data

# In[2]:


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=';')


# In[3]:


data


# In[4]:


X = data.drop("quality", axis=1)
y = data["quality"]


# In[5]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# # B) Train Model and MLFlow

# In[7]:


import mlflow
import mlflow.sklearn


# In[8]:


experiment_name = "Wine_Quality_Classification"
mlflow.set_experiment(experiment_name)


# In[13]:


with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.sklearn.log_model(model, "random_forest_classifier")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", metric_value)
        else:
            mlflow.log_metric(label, metrics)
    
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.barh(pos, feature_importances[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.savefig('feature_importances.png')
    
    mlflow.log_artifact('feature_importances.png')


# In[ ]:




# In[21]:


def preprocess_data():    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, delimiter=";")
    X = data.drop("quality", axis=1)
    y = data["quality"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


# In[22]:


def train_model(X_train, y_train):
    experiment_name = "Wine_Quality_Classification"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.sklearn.log_model(model, "random_forest_classifier")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", metric_value)
            else:
                mlflow.log_metric(label, metrics)

        feature_importances = model.feature_importances_
        sorted_idx = np.argsort(feature_importances)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        plt.barh(pos, feature_importances[sorted_idx], align='center')
        plt.yticks(pos, X.columns[sorted_idx])
        plt.xlabel('Importance')
        plt.title('Feature Importances')
        plt.savefig('feature_importances.png')

        mlflow.log_artifact('feature_importances.png')
    
    return model


# In[23]:


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


# In[24]:


def deploy_model(model):
    get_ipython().system('git push')
    pass


# In[25]:


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 5, 5),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# In[26]:


dag = DAG(
    "wine_quality_pipeline",
    default_args=default_args,
    description="A pipeline to preprocess, train, evaluate, and deploy a wine quality prediction model",
    schedule_interval=timedelta(days=1),
    catchup=False,
)


# In[33]:


preprocess_task = PythonOperator(
    task_id="preprocess_data2",
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_model2",
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id="evaluate_model2",
    python_callable=evaluate_model,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id="deploy_model2",
    python_callable=deploy_model,
    dag=dag,
)


# In[34]:


preprocess_task >> train_task
train_task >> evaluate_task
evaluate_task >> deploy_task


# In[ ]:




