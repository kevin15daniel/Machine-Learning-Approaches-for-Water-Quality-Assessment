# -*- coding: utf-8 -*-
"""Machine Learning Approaches for Water Quality Assessment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1x5zOxxfgjIG_vwsPKKeTWRTpfx25BZKt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

"""### **Data Preparation:**"""

from google.colab import files

uploaded = files.upload()

data = pd.read_csv('water_classification_model.csv')
df = pd.DataFrame(data)
df.head()

df.shape

df.info()

df.describe()

df.nunique()

df['Potability'].value_counts()

sns.set(style='whitegrid')

sns.countplot(data=df, x='Potability', palette='viridis', edgecolor='black').set(title="WATER POTABILITY DISTRIBUTION")

sns.set(style='whitegrid')

ax = df['Potability'].value_counts().plot(kind='pie',
                                           colors=sns.color_palette('viridis', n_colors=len(df['Potability'].value_counts())),
                                           figsize=(6, 6),
                                           title="WATER POTABILITY DISTRIBUTION",
                                           autopct='%1.1f%%')
ax.set_ylabel('')
for wedge in ax.patches:
    wedge.set_edgecolor('black')

plt.show()

non_potable = df[df['Potability'] == 0]
potable = df[df['Potability'] == 1]

fig, axes = plt.subplots(3, 3, figsize=(13, 13))
axes = axes.flatten()

for ax, col in zip(axes, df.columns[:9]):
    ax.set_title(col)
    sns.kdeplot(non_potable[col], label='Non Potable', ax=ax)
    sns.kdeplot(potable[col], label='Potable', ax=ax)
    ax.legend()

plt.suptitle("WATER QUALITY DISTRIBUTION", y=1.01, size=16, color='black')
plt.tight_layout()

df.drop('Potability', axis=1).skew()

df.corr()

fig = px.imshow(df.corr(),
                color_continuous_scale='RdBu',
                title="WATER QUALITY HEAT MAP",
                labels=dict(x='Features (x)', y='Features (y)', color='corr'))

fig.update_layout(title_font_size=16)

fig.show()

df.corr().abs()['Potability'].sort_values(ascending=False)

ax = sns.pairplot(df, hue='Potability', diag_kind='kde', kind='scatter')
ax.fig.set_size_inches(16, 16)
ax.fig.suptitle("WATER QUALITY PAIR-PLOT", y=1.01, size=16, color='black')

fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(30, 10))
for index, attributeCol in enumerate(df.columns):
    sns.boxplot(y=attributeCol, data=df, ax=ax.flatten()[index])

plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=5.0)
plt.suptitle("WATER QUALITY BOX-PLOT", y=1.01, size=16, color='black')

"""### **Handling Missing Data:**"""

df.isna().any().any()

df.isnull().sum()

missing_data = df.isnull().mean() * 100
fig = px.bar(missing_data,
             title="MISSING DATA IN PERCENTAGES",
             labels={'index': 'Features', 'value': 'Percentage of Missing Values'},
             color=missing_data,
             color_continuous_scale='RdBu')

fig.update_layout(
    xaxis_title='Features (x)',
    yaxis_title='Percentage of Missing Values (y)',
    title_font_size=16,
)

fig.show()

df.isnull().mean() * 100

df[df['Potability'] == 0][['ph', 'Sulfate', 'Trihalomethanes']].mean()

df[df['Potability'] == 1][['ph', 'Sulfate', 'Trihalomethanes']].mean()

df1 = df.copy()
df1[['ph', 'Sulfate', 'Trihalomethanes']] = df1[['ph', 'Sulfate', 'Trihalomethanes']].fillna(df1[['ph', 'Sulfate', 'Trihalomethanes']].mean())

df1.isnull().sum()

"""### **Handling Anomalies:**"""

from scipy import stats

df_water = df1.copy()
df_water = df_water[(np.abs(stats.zscore(df_water)) <= 3).all(axis=1)]

df_water.shape

fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(30, 10))
index = 0
ax = ax.flatten()

for attributeCol, value in df_water.items():
    sns.boxplot(y=attributeCol, data=df_water, ax=ax[index])
    index += 1

plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=5.0)
plt.suptitle("WATER QUALITY BOX-PLOT (POST OUTLIER REMOVAL)", y=1.01, size=16, color='black')

"""### **Handling Class Imbalance:**"""

count_class_0, count_class_1 = df_water.Potability.value_counts()

df_class_0 = df_water[df_water['Potability'] == 0]
df_class_1 = df_water[df_water['Potability'] == 1]

sns.set(style='whitegrid')

df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print("RANDOM OVER-SAMPLING:")
print(df_test_over.Potability.value_counts())

ax = df_test_over['Potability'].value_counts().plot(kind='pie',
                                                   colors=sns.color_palette('viridis', n_colors=len(df_test_over['Potability'].value_counts())),
                                                   figsize=(6, 6),
                                                   title="COUNT (TARGET)",
                                                   autopct='%1.1f%%')
ax.set_ylabel('')
for wedge in ax.patches:
    wedge.set_edgecolor('black')

plt.show()

"""### **Analyzing Correlation using ANOVA:**"""

def FunctionAnova(inpData, TargetVariable, attributeList):
    from scipy.stats import f_oneway
    SelectedPredictors = []

    for predictor in attributeList:
        CategoryGroupLists = inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)

        if AnovaResults[1] < 0.05:
            print(f"{predictor} is correlated with {TargetVariable}, P-Value: {AnovaResults[1]}")
            SelectedPredictors.append(predictor)
        else:
            print(f"{predictor} is NOT correlated with {TargetVariable}, P-Value: {AnovaResults[1]}")

    return SelectedPredictors

attributeColList = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

FunctionAnova(
    inpData=df_water,
    TargetVariable='Potability',
    attributeList=attributeColList
)

"""### **Data Preparation - Splitting and Scaling:**"""

from sklearn.model_selection import train_test_split

X = df_test_over.drop('Potability', axis=1)
y = df_test_over['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

print('X_train:', X_train.shape, '\ny_train:', y_train.shape)
print('X_test:', X_test.shape, '\ny_test:', y_test.shape)

"""### **Dimensionality Reduction through PCA:**"""

from sklearn.decomposition import PCA

pca = PCA()

X_train_pca = pca.fit_transform(X_train)
exp_var_pca = pca.explained_variance_ratio_

cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual Explained Variance')
plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid', label='Cumulative Explained Variance')
plt.xlabel('Principal Component Index (x)')
plt.ylabel('Explained Variance Ratio (y)')
plt.title("WATER QUALITY USING PCA")
plt.legend(loc='best')
plt.show()

"""### **Model Training - First Iteration:**"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score, accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, classification_report, recall_score, f1_score

"""### 1. **Logistic Regression - First Iteration:**"""

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

LogReg_pred = LogReg.predict(X_test)

LogReg_acc = accuracy_score(y_test, LogReg_pred)
LogReg_mae = mean_absolute_error(y_test, LogReg_pred)
LogReg_mse = mean_squared_error(y_test, LogReg_pred)
LogReg_rmse = np.sqrt(LogReg_mse)

LogReg_precision = precision_score(y_test, LogReg_pred)
LogReg_recall = recall_score(y_test, LogReg_pred)
LogReg_f1 = f1_score(y_test, LogReg_pred)

print("The Accuracy for Logistic Regression is", LogReg_acc)

print("The Classification Report using Logistic Regression is:")
print(classification_report(y_test, LogReg_pred))

LogReg_cm = confusion_matrix(y_test, LogReg_pred)
sns.heatmap(LogReg_cm / np.sum(LogReg_cm), annot=True, fmt='0.2%', cmap='RdBu')
plt.title("CONFUSION MATRIX FOR LOGISTIC REGRESSION")

"""### 2. **K-Nearest Neighbors (KNN) Regression - First Iteration:**"""

KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)

KNN_pred = KNN.predict(X_test)

KNN_acc = accuracy_score(y_test, KNN_pred)
KNN_mae = mean_absolute_error(y_test, KNN_pred)
KNN_mse = mean_squared_error(y_test, KNN_pred)
KNN_rmse = np.sqrt(KNN_mse)

KNN_precision = precision_score(y_test, KNN_pred)
KNN_recall = recall_score(y_test, KNN_pred)
KNN_f1 = f1_score(y_test, KNN_pred)

print("The Accuracy for K-Nearest Neighbors (KNN) is", KNN_acc)

print("The Classification Report using K-Nearest Neighbors (KNN) is:")
print(classification_report(y_test, KNN_pred))

KNN_cm = confusion_matrix(y_test, KNN_pred)
sns.heatmap(KNN_cm / np.sum(KNN_cm), annot=True, fmt='0.2%', cmap='RdBu')
plt.title("CONFUSION MATRIX FOR KNN")

"""### 3. **Decision Tree Classifier - First Iteration:**"""

DecTree = DecisionTreeClassifier()
DecTree.fit(X_train, y_train)

DecTree_pred = DecTree.predict(X_test)

DecTree_acc = accuracy_score(y_test, DecTree_pred)
DecTree_precision = precision_score(y_test, DecTree_pred)
DecTree_recall = recall_score(y_test, DecTree_pred)
DecTree_f1 = f1_score(y_test, DecTree_pred)

print("The Accuracy for Decision Tree is", DecTree_acc)

print("The Classification Report using Decision Tree is:")
print(classification_report(y_test, DecTree_pred))

DecTree_cm = confusion_matrix(y_test, DecTree_pred)
sns.heatmap(DecTree_cm / np.sum(DecTree_cm), annot=True, fmt='0.2%', cmap='RdBu')
plt.title("CONFUSION MATRIX FOR DECISION TREE")

"""### 4. **Random Forest Classifier - First Iteration:**"""

RFTree = RandomForestClassifier()
RFTree.fit(X_train, y_train)

RFTree_pred = RFTree.predict(X_test)

RFTree_acc = accuracy_score(y_test, RFTree_pred)
RFTree_precision = precision_score(y_test, RFTree_pred)
RFTree_recall = recall_score(y_test, RFTree_pred)
RFTree_f1 = f1_score(y_test, RFTree_pred)

print("The Accuracy for Random Forest is", RFTree_acc)

print("The Classification Report using Random Forest is:")
print(classification_report(y_test, RFTree_pred))

RFTree_cm = confusion_matrix(y_test, RFTree_pred)
sns.heatmap(RFTree_cm / np.sum(RFTree_cm), annot=True, fmt='0.2%', cmap='RdBu')
plt.title("CONFUSION MATRIX FOR RANDOM FOREST")

"""### 5. **Support Vector Machine (SVM) Classifier - First Iteration:**"""

SVM = SVC()
SVM.fit(X_train, y_train)

SVM_pred = SVM.predict(X_test)

SVM_acc = accuracy_score(y_test, SVM_pred)
SVM_precision = precision_score(y_test, SVM_pred)
SVM_recall = recall_score(y_test, SVM_pred)
SVM_f1 = f1_score(y_test, SVM_pred)

print("The Accuracy for Support Vector Machine (SVM) is", SVM_acc)

print("The Classification Report using Support Vector Machine (SVM) is:")
print(classification_report(y_test, SVM_pred))

SVM_cm = confusion_matrix(y_test, SVM_pred)
sns.heatmap(SVM_cm / np.sum(SVM_cm), annot=True, fmt='0.2%', cmap='RdBu')
plt.title("CONFUSION MATRIX FOR SVM")

"""### 6. **XGBoost Classifier - First Iteration:**"""

XGB = XGBClassifier()
XGB.fit(X_train, y_train)

XGB_pred = XGB.predict(X_test)

XGB_acc = accuracy_score(y_test, XGB_pred)
XGB_precision = precision_score(y_test, XGB_pred)
XGB_recall = recall_score(y_test, XGB_pred)
XGB_f1 = f1_score(y_test, XGB_pred)

print("The Accuracy for XGBoost is", XGB_acc)

print("The Classification Report using XGBoost is:", XGB_acc)
print(classification_report(y_test, XGB_pred))

XGB_cm = confusion_matrix(y_test, XGB_pred)
sns.heatmap(XGB_cm / np.sum(XGB_cm), annot=True, fmt='0.2%', cmap='RdBu')
plt.title("CONFUSION MATRIX FOR XGBOOST")

"""### **Model Comparison - First Iteration:**"""

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'XGBoost'],
    'Accuracy': [LogReg_acc, KNN_acc, DecTree_acc, RFTree_acc, SVM_acc, XGB_acc],
    'Precision': [LogReg_precision, KNN_precision, DecTree_precision, RFTree_precision, SVM_precision, XGB_precision],
    'Recall': [LogReg_recall, KNN_recall, DecTree_recall, RFTree_recall, SVM_recall, XGB_recall],
    'F1 Score': [LogReg_f1, KNN_f1, DecTree_f1, RFTree_f1, SVM_f1, XGB_f1]
})

models = models.sort_values(by='Accuracy', ascending=False)

models

fig = px.bar(models,
             x='Accuracy',
             y='Model',
             color='Model',
             color_continuous_scale='RdBu',
             title="INITIAL EVALUATION AND COMPARISON OF MODEL(s)")

fig.update_layout(
    title_x=0.5,
    xaxis_title='Accuracy (x)',
    yaxis_title='Model (y)',
    xaxis=dict(range=[0, 1])
)

fig.show()

"""### **Model Tuning:**"""

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

lgr = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svc = SVC()
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)

para_lgr = {'solver': ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none']}

grid_lgr = GridSearchCV(lgr, param_grid=para_lgr, cv=5)
grid_lgr.fit(X_train, y_train)

print("Optimal hyperparameters for Logistic Regression:", grid_lgr.best_params_)

para_knn = {'n_neighbors': np.arange(1, 50),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knn = GridSearchCV(knn, param_grid=para_knn, cv=5)
grid_knn.fit(X_train, y_train)

print("Optimal hyperparameters for KNN:", grid_knn.best_params_)

para_dt = {'criterion': ['gini', 'entropy'],
           'max_depth': np.arange(1, 50),
           'min_samples_leaf': [1, 2, 4, 5, 10, 20, 30, 40, 80, 100]}

grid_dt = GridSearchCV(dt, param_grid=para_dt, cv=5)
grid_dt.fit(X_train, y_train)

print("Optimal hyperparameters for Decision Tree:", grid_dt.best_params_)

params_rf = {'n_estimators': [100, 200, 350, 500],
             'min_samples_leaf': [2, 10, 30]}

grid_rf = GridSearchCV(rf, param_grid=params_rf, cv=5)
grid_rf.fit(X_train, y_train)

print("Optimal hyperparameters for Random Forest:", grid_rf.best_params_)

para_svc = {'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']}

grid_svc = GridSearchCV(svc, param_grid=para_svc, cv=5)
grid_svc.fit(X_train, y_train)

print("Optimal hyperparameters for SVM:", grid_svc.best_params_)

params_xgb = {'n_estimators': [50, 100, 250, 400, 600, 800, 1000],
              'learning_rate': [0.2, 0.5, 0.8, 1]}

rs_xgb = RandomizedSearchCV(xgb, param_distributions=params_xgb, cv=5)
rs_xgb.fit(X_train, y_train)

print("Optimal hyperparameters for XGBoost:", rs_xgb.best_params_)

"""### **Post-Hyperparameter Tuning and Model Refinement:**

### 1. **Logistic Regression - Second Iteration:**
"""

LogReg2 = LogisticRegression(penalty='l1', solver='liblinear')
LogReg2.fit(X_train, y_train)

LogReg2_pred = LogReg2.predict(X_test)

LogReg2_acc = accuracy_score(y_test, LogReg2_pred)
LogReg2_mae = mean_absolute_error(y_test, LogReg2_pred)
LogReg2_mse = mean_squared_error(y_test, LogReg2_pred)
LogReg2_rmse = np.sqrt(LogReg2_mse)

LogReg2_precision = precision_score(y_test, LogReg2_pred)
LogReg2_recall = recall_score(y_test, LogReg2_pred)
LogReg2_f1 = f1_score(y_test, LogReg2_pred)

print("The Accuracy for Logistic Regression is", LogReg2_acc)

print("The Classification Report using Logistic Regression is:")
print(classification_report(y_test, LogReg2_pred))

LogReg2_cm = confusion_matrix(y_test, LogReg2_pred)
sns.heatmap(LogReg2_cm / np.sum(LogReg2_cm), annot=True, fmt='0.2%', cmap='Blues')
plt.title("CONFUSION MATRIX FOR LOGISTIC REGRESSION")

"""### 2. **K-Nearest Neighbors (KNN) Regression - Second Iteration:**"""

KNN2 = KNeighborsClassifier(algorithm='auto', n_neighbors=1, weights='uniform')
KNN2.fit(X_train, y_train)

KNN2_pred = KNN2.predict(X_test)

KNN2_acc = accuracy_score(y_test, KNN2_pred)
KNN2_mae = mean_absolute_error(y_test, KNN2_pred)
KNN2_mse = mean_squared_error(y_test, KNN2_pred)
KNN2_rmse = np.sqrt(mean_squared_error(y_test, KNN2_pred))

KNN2_precision = precision_score(y_test, KNN2_pred)
KNN2_recall = recall_score(y_test, KNN2_pred)
KNN2_f1 = f1_score(y_test, KNN2_pred)

print("The Accuracy for K-Nearest Neighbors (KNN) is", KNN2_acc)

print("The Classification Report using K-Nearest Neighbors (KNN) is:", KNN2_acc)
print(classification_report(y_test, KNN2_pred))

KNN2_cm = confusion_matrix(y_test, KNN2_pred)
sns.heatmap(KNN2_cm / np.sum(KNN2_cm), annot=True, fmt='0.2%', cmap='Blues')
plt.title("CONFUSION MATRIX FOR KNN")

"""### 3. **Decision Tree Classifier - Second Iteration:**"""

DecTree2 = DecisionTreeClassifier(criterion='entropy', max_depth=44, min_samples_leaf=1)
DecTree2.fit(X_train, y_train)

DecTree2_pred = DecTree2.predict(X_test)

DecTree2_acc = accuracy_score(y_test, DecTree2_pred)
DecTree2_precision = precision_score(y_test, DecTree2_pred)
DecTree2_recall = recall_score(y_test, DecTree2_pred)
DecTree2_f1 = f1_score(y_test, DecTree2_pred)

print("The Accuracy for Decision Tree is", DecTree2_acc)

print("The Classification Report using Decision Tree is:")
print(classification_report(y_test, DecTree2_pred))

DecTree2_cm = confusion_matrix(y_test, DecTree2_pred)
sns.heatmap(DecTree2_cm / np.sum(DecTree2_cm), annot=True, fmt='0.2%', cmap='Blues')
plt.title("CONFUSION MATRIX FOR DECISION TREE")

"""### 4. **Random Forest Classifier - Second Iteration**:"""

RFTree2 = RandomForestClassifier(min_samples_leaf=2, n_estimators=200)
RFTree2.fit(X_train, y_train)

RFTree2_pred = RFTree2.predict(X_test)

RFTree2_acc = accuracy_score(y_test, RFTree2_pred)
RFTree2_precision = precision_score(y_test, RFTree2_pred)
RFTree2_recall = recall_score(y_test, RFTree2_pred)
RFTree2_f1 = f1_score(y_test, RFTree2_pred)

print("The Accuracy for Random Forest is", RFTree2_acc)

print("The Classification Report using Random Forest is:")
print(classification_report(y_test, RFTree2_pred))

RFTree2_cm = confusion_matrix(y_test, RFTree2_pred)
sns.heatmap(RFTree2_cm / np.sum(RFTree2_cm), annot=True, fmt='0.2%', cmap='Blues')
plt.title("CONFUSION MATRIX FOR RANDOM FOREST")

"""### 5. **Support Vector Machine (SVM) Classifier - Second Iteration:**"""

SVM2 = SVC(C=10, gamma=1, kernel='rbf')
SVM2.fit(X_train, y_train)

SVM2_pred = SVM2.predict(X_test)

SVM2_acc = accuracy_score(y_test, SVM2_pred)
SVM2_precision = precision_score(y_test, SVM2_pred)
SVM2_recall = recall_score(y_test, SVM2_pred)
SVM2_f1 = f1_score(y_test, SVM2_pred)

print("The Accuracy for Support Vector Machine (SVM) is", SVM2_acc)

print("The Classification Report using Support Vector Machine (SVM) is:", SVM2_acc)
print(classification_report(y_test, SVM2_pred))

SVM2_cm = confusion_matrix(y_test, SVM2_pred)
sns.heatmap(SVM2_cm / np.sum(SVM2_cm), annot=True, fmt='0.2%', cmap='Blues')
plt.title("CONFUSION MATRIX FOR SVM")

"""### 6. **XGBoost Classifier - Second Iteration:**"""

XGB2 = XGBClassifier(n_estimators=600, learning_rate=0.8)
XGB2.fit(X_train, y_train)

XGB2_pred = XGB2.predict(X_test)

XGB2_acc = accuracy_score(y_test, XGB2_pred)
XGB2_precision = precision_score(y_test, XGB2_pred)
XGB2_recall = recall_score(y_test, XGB2_pred)
XGB2_f1 = f1_score(y_test, XGB2_pred)

print("The Accuracy for XGBoost is", XGB2_acc)

print("The Classification Report using XGBoost is:", XGB2_acc)
print(classification_report(y_test, XGB2_pred))

XGB2_cm = confusion_matrix(y_test, XGB2_pred)
sns.heatmap(XGB2_cm / np.sum(XGB2_cm), annot=True, fmt='0.2%', cmap='Blues')
plt.title("CONFUSION MATRIX FOR XGBOOST")

"""### **Model Comparison - Second Iteration:**"""

models2 = pd.DataFrame({
    'Model': ['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'XGBoost'],
    'Accuracy': [LogReg2_acc, KNN2_acc, DecTree2_acc, RFTree2_acc, SVM2_acc, XGB2_acc],
    'Precision': [LogReg2_precision, KNN2_precision, DecTree2_precision, RFTree2_precision, SVM2_precision, XGB2_precision],
    'Recall': [LogReg2_recall, KNN2_recall, DecTree2_recall, RFTree2_recall, SVM2_recall, XGB2_recall],
    'F1 Score': [LogReg2_f1, KNN2_f1, DecTree2_f1, RFTree2_f1, SVM2_f1, XGB2_f1]
})

models2 = models2.sort_values(by='Accuracy', ascending=False)

models2

fig = px.bar(models2,
             x='Accuracy',
             y='Model',
             color='Model',
             color_continuous_scale='Blues',
             title="FINAL EVALUATION AND COMPARISON OF MODEL(s)")

fig.update_layout(
    title_x=0.5,
    xaxis_title='Accuracy (x)',
    yaxis_title='Model (y)',
    xaxis=dict(range=[0, 1])
)

fig.show()

"""### **Performance Comparison - Iteration 1 vs Iteration 2:**"""

comp_iterations = pd.DataFrame({
    'Model': ['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'XGBoost'],
    'Iteration 1': [LogReg_acc, KNN_acc, DecTree_acc, RFTree_acc, SVM_acc, XGB_acc],
    'Iteration 2': [LogReg2_acc, KNN2_acc, DecTree2_acc, RFTree2_acc, SVM2_acc, XGB2_acc]
})

comp_iterations

fig = px.bar(comp_iterations,
             x=['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Support Vector Machine	', 'XGBoost'],
             y=comp_iterations.columns[1:],
             labels={'x': 'Model', 'y': 'Percentage of Accuracy'},
             title="COMPARISON BETWEEN ITERATIONS")

fig.update_layout(barmode='group')

fig.show()

"""### **Cross Validation:**"""

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from statistics import mean, stdev

cv = KFold(n_splits=10, random_state=1, shuffle=True)

RFTree2_scores = cross_val_score(RFTree2, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
SVM2_scores = cross_val_score(SVM2, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
XGB2_scores = cross_val_score(XGB2, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print("Accuracy of CV - Random Forest: %.4f (%.4f)" % (mean(RFTree2_scores), stdev(RFTree2_scores)))
print("Accuracy of CV - SVM: %.4f (%.4f)" % (mean(SVM2_scores), stdev(SVM2_scores)))
print("Accuracy of CV - XGBoost: %.4f (%.4f)" % (mean(XGB2_scores), stdev(XGB2_scores)))

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, SVM2_pred)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.rcParams['font.size'] = 12
plt.title("ROC CURVE FOR SVM IN WATER QUALITY")
plt.xlabel('False Positive Rate (x)')
plt.ylabel('True Positive Rate (y)')
plt.show()

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, SVM2_pred)
print("ROC AUC: {:.4f}".format(ROC_AUC))

Cross_validated_ROC_AUC = cross_val_score(SVM2, X_train, y_train, cv=10, scoring='roc_auc').mean()
print("Cross-validated ROC AUC: {:.4f}".format(Cross_validated_ROC_AUC))