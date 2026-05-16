import pandas as pd
import numpy as np
import json
import joblib
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# load & prepare data 

df = pd.read_csv('medical_appointment.csv')

df.rename(columns={
    'Hipertension': 'Hypertension',
    'Handcap': 'Handicap',
    'No-show': 'NoShow'
}, inplace=True)

df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

df['NoShow'] = df['NoShow'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

df['WaitingDays'] = (
    df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()
).dt.days

df = df[df['Age'] >= 0]
df = df[df['WaitingDays'] >= 0]
df.reset_index(drop=True, inplace=True)

df['ScheduledHour']        = df['ScheduledDay'].dt.hour
df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.dayofweek

df = df.sort_values('AppointmentDay').reset_index(drop=True)
df['PreviousNoShowRate'] = df.groupby('PatientId')['NoShow'].transform(
    lambda x: x.shift().expanding().mean()
).fillna(0)

bins   = [-1, 12, 17, 35, 55, 120]
labels = ['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

df['HasCondition'] = (
    (df['Hypertension'] == 1) | (df['Diabetes'] == 1) | (df['Alcoholism'] == 1)
).astype(int)

top = df['Neighbourhood'].value_counts()[df['Neighbourhood'].value_counts() >= 100].index
df['Neighbourhood'] = df['Neighbourhood'].apply(lambda x: x if x in top else 'Other')

df['WaitBucket'] = pd.cut(
    df['WaitingDays'],
    bins=[-1, 0, 7, 30, 60, 180],
    labels=['Same Day', '1-7', '8-30', '31-60', '60+']
)

df.drop(columns=['ScheduledDay', 'AppointmentDay', 'WaitBucket'], inplace=True)

# train test split 

X = df.drop(columns=['NoShow'])
y = df['NoShow']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  preprocessor 

num_cols = ['Age', 'WaitingDays', 'ScheduledHour', 'PreviousNoShowRate', 'AppointmentDayOfWeek']
cat_cols = ['Neighbourhood', 'AgeGroup']
bin_cols = ['Gender', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received', 'HasCondition']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OrdinalEncoder(), cat_cols),
    ('bin', 'passthrough', bin_cols)
])

#  model comparison
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(random_state=42, eval_metric='logloss')
}

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote',         SMOTE(random_state=42)),
        ('selector',      SelectKBest(f_classif, k=10)),
        ('model',         model)
    ])

    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=['roc_auc', 'f1', 'precision', 'recall'],
                            n_jobs=-1)

    results.append({
        'Model':     name,
        'AUC-ROC':   round(scores['test_roc_auc'].mean(), 4),
        'F1':        round(scores['test_f1'].mean(), 4),
        'Precision': round(scores['test_precision'].mean(), 4),
        'Recall':    round(scores['test_recall'].mean(), 4)
    })
    print(f'{name} done')

results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
print(results_df)

# hyperparameter tuning 

scale = (y_train == 0).sum() / (y_train == 1).sum()

xgb_pipe = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote',        SMOTE(random_state=42)),
    ('selector',     SelectKBest(f_classif, k=10)),
    ('model',        XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=round(scale)))
])

param_grid = {
    'model__n_estimators':  [100, 200],
    'model__max_depth':     [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample':     [0.8, 1.0]
}

grid = GridSearchCV(
    xgb_pipe, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print('Best params:', grid.best_params_)
print('Best CV AUC:', round(grid.best_score_, 4))

best_model = grid.best_estimator_

#  evaluate on test set 

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    'AUC_ROC':   round(roc_auc_score(y_test, y_prob), 4),
    'F1':        round(f1_score(y_test, y_pred), 4),
    'Precision': round(precision_score(y_test, y_pred), 4),
    'Recall':    round(recall_score(y_test, y_pred), 4),
    'Accuracy':  round(accuracy_score(y_test, y_pred), 4)
}

print(test_metrics)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#  save model 

joblib.dump(best_model, 'noshow_model.pkl')
print('Model saved')

# save results for streamlit

ml_results = {
    'model_comparison': results_df.to_dict(orient='records'),
    'best_params':      grid.best_params_,
    'cv_best_auc':      round(grid.best_score_, 4),
    'test_metrics':     test_metrics,
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    'classification_report': classification_report(y_test, y_pred, output_dict=True)
}

with open('ml_results.json', 'w') as f:
    json.dump(ml_results, f)

print('Results saved to ml_results.json')
