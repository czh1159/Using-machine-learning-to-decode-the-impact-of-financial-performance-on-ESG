
#ALEP & ICEP
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
#Assume that Column 14 is an important feature for Y
def plot_rf(data):
    X1 = data.drop(['ESG','code','year'], axis=1)
    y = data['ESG']
    X = X1.dropna()
    y = y[X.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X=pd.DataFrame(X)  
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=2)
    rf_model = RandomForestRegressor(n_estimators=200,max_depth=6,random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"均方误差: {mse}")
    feature_names = [14]
    # ALE plot
    for feature in feature_names:
        PartialDependenceDisplay.from_estimator(rf_model, X, features=[feature], kind='average')
        plt.title(f'ALE 图 - {X1.columns[feature]} 对ESG水平的影响')
        plt.show()
    for feature in feature_names:
        PartialDependenceDisplay.from_estimator(rf_model, X, features=[feature], kind='individual')
        plt.title(f'ICE 图 - {X1.columns[feature]} 对ESG水平的影响')
        plt.show()
plot_rf(data)

def plot_xgb(data):
    X1 = data.drop(['ESG','code','year','G'], axis=1)
    y = data['G']
    X = X1.dropna()
    y = y[X.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X=pd.DataFrame(X)  
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=2)
    xgb_model = xgb.XGBRegressor(n_estimators=200, random_state=0)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"均方误差: {mse}")
    feature_names = [5,6]
    for feature in feature_names:
        PartialDependenceDisplay.from_estimator(xgb_model, X, features=[feature], kind='average')
        plt.title(f'ALE 图 - {X1.columns[feature]} 对 ESG 水平的影响')
        plt.show()
    for feature in feature_names:
        PartialDependenceDisplay.from_estimator(xgb_model, X, features=[feature], kind='individual')
        plt.title(f'ICE 图 - {X1.columns[feature]} 对 ESG 水平的影响')
        plt.show()
plot_xgb(data)



















