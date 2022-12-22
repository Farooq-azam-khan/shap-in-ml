# https://www.kdnuggets.com/2022/11/shap-explain-machine-learning-model-python.html
import pandas as pd 

data = pd.read_csv('advertising.csv')
data.columns = data.columns.map(lambda row: '_'.join(row.lower().split(' ')))

print('data shape = ', data.shape)
print(data.describe())
print(data.head())

from patsy import dmatrices 
y, X = dmatrices(
        'clicked_on_ad ~ daily_time_spent_on_site + age + area_income + daily_internet_usage + male - 1'
        , data = data 
)
X_frame = pd.DataFrame(data=X, columns=X.design_info.column_names)
print(X_frame.shape)
print(X_frame.head())

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

import xgboost 
print('Training Model...')
model = xgboost.XGBClassifier().fit(X_train, y_train)
predict = model.predict(X_test)

from sklearn.metrics import f1_score 
f1 = f1_score(y_test, predict)
print('f1:', f1)

