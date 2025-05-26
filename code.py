import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
#RF
def observe_performance_rf(data):
    X = data.drop(['ESG', 'code', 'year'], axis=1)
    y = data['ESG']
    X = X.dropna()
    y = y[X.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    rf_model = RandomForestRegressor(n_estimators=150,random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"样本外均方误差: {mse}")
    y_pred_train = rf_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print(f"样本内拟合优度（训练集）: {r2_train}")
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均绝对误差（MAE）: {mae}")
    medae = np.median(np.abs(y_test - y_pred))
    print(f"绝对中位差（MedAE）: {medae}")
    explained_variance = explained_variance_score(y_test, y_pred)
    print(f"可解释方差: {explained_variance}")
#DT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
def observe_performance_dt(data):
    X = data.drop(['ESG', 'code', 'year'], axis=1)
    y = data['ESG']
    X = X.dropna()
    y = y[X.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=0)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"样本外均方误差: {mse}")
    y_pred_train = dt_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print(f"样本内拟合优度（训练集）: {r2_train}")
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均绝对误差（MAE）: {mae}")
    medae = np.median(np.abs(y_test - y_pred))
    print(f"绝对中位差（MedAE）: {medae}")
    explained_variance = explained_variance_score(y_test, y_pred)
    print(f"可解释方差: {explained_variance}")
#SVM 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
def observe_performance_svm(data):
    X = data.drop(['ESG', 'code', 'year'], axis=1)
    y = data['ESG']
    X = X.dropna()
    y = y[X.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"样本外均方误差: {mse}")
    y_pred_train = svm_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print(f"样本内拟合优度（训练集）: {r2_train}")
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均绝对误差（MAE）: {mae}")
    medae = np.median(np.abs(y_test - y_pred))
    print(f"绝对中位差（MedAE）: {medae}")
    explained_variance = explained_variance_score(y_test, y_pred)
    print(f"可解释方差: {explained_variance}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均绝对误差（MAE）: {mae}")
    medae = np.median(np.abs(y_test - y_pred))
    print(f"绝对中位差（MedAE）: {medae}")
    explained_variance = explained_variance_score(y_test, y_pred)
    print(f"可解释方差: {explained_variance}")
#KNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
def observe_performance_knn(data):
    X = data.drop(['ESG', 'code', 'year'], axis=1)
    y = data['ESG']
    X = X.dropna()
    y = y[X.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    knn_model = KNeighborsRegressor(n_neighbors=120)  # 可以调整 n_neighbors 的值
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"样本外均方误差: {mse}")
    y_pred_train = knn_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print(f"样本内拟合优度（训练集）: {r2_train}")
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均绝对误差（MAE）: {mae}")
    medae = np.median(np.abs(y_test - y_pred))
    print(f"绝对中位差（MedAE）: {medae}")
    explained_variance = explained_variance_score(y_test, y_pred)
    print(f"可解释方差: {explained_variance}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均绝对误差（MAE）: {mae}")
    medae = np.median(np.abs(y_test - y_pred))
    print(f"绝对中位差（MedAE）: {medae}")
    explained_variance = explained_variance_score(y_test, y_pred)
    print(f"可解释方差: {explained_variance}")
#XGB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def observe_performance_xgb(data):
    X = data.drop(['ESG', 'code', 'year'], axis=1)
    y = data['ESG']
    X = X.dropna()
    y = y[X.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    xgb_model = xgb.XGBRegressor(n_estimators=150, _state=0)
    xgb_model.fit(X_train, y_train)
    y_pred = gbrt_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"样本外均方误差: {mse}")
    y_pred_train = gbrt_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print(f"样本内拟合优度（训练集）: {r2_train}")
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均绝对误差（MAE）: {mae}")
    medae = np.median(np.abs(y_test - y_pred))
    print(f"绝对中位差（MedAE）: {medae}")
    explained_variance = explained_variance_score(y_test, y_pred)
    print(f"可解释方差: {explained_variance}")

#LASSO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
def observe_performance_lasso(data):
    X = data.drop(['ESG', 'code', 'year'], axis=1)
    y = data['ESG']
    X = X.dropna()
    y = y[X.index]
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    lasso_model = Lasso(alpha=0.01) 
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"样本外均方误差: {mse}")
    y_pred_train = lasso_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print(f"样本内拟合优度（训练集）: {r2_train}")
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均绝对误差（MAE）: {mae}")
    medae = np.median(np.abs(y_test - y_pred))
    print(f"绝对中位差（MedAE）: {medae}")
    explained_variance = explained_variance_score(y_test, y_pred)
    print(f"可解释方差: {explained_variance}")
#data reading
###The data of different dimensions are merged by permutation and combination, and the above functions are called for calculation.
df1=pd.read_csv('D:/esg/B_dimension_esg.csv', encoding='ANSI')
df2=pd.read_csv('D:/esg/O_dimension_esg.csv', encoding='ANSI')
df3=pd.read_csv('D:/esg/D_dimension_esg.csv', encoding='ANSI')
df4=pd.read_csv('D:/esg/P_dimension_esg.csv', encoding='ANSI')
df5=pd.read_csv('D:/esg/G_dimension_esg.csv', encoding='ANSI')
#The experimental results here correspond to Tables 2 and 3
#You need to merge the data according to your file name.
#e.g  Combining the data of B and G dimensions as df_B_G.
observe_performance_rf(df_B_G)
observe_performance_knn(df_B_G)
observe_performance_lasso(df_B_G)
observe_performance_xgb(df_B_G)
observe_performance_dt(df_B_G)


###festure_importance  Table4 & Table 5
def importance_rf(data):
    aa=0
    X1 = data.drop(['ESG','code','year'], axis=1)
    y = data['ESG']
    X = X1.dropna()
    y = y[X.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X=pd.DataFrame(X)  
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=2)
    rf_model = RandomForestRegressor(n_estimators=150, random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"均方误差: {mse}")
    feature_names = [6,7,8,9,10,11,12,13,14,15]#O
    feature_names2 = [15,16,17,18,19,20,21,22,23,24]#D
    feature_names3 = [25,26,27,28,29,30,31,32,33,34]#P
    feature_names4 = [35,36,37,38,39,40,41,42,43,44]#G
    importances = rf_model.feature_importances_
    for feature in feature_names2:
        print(X1.columns[feature])
        print(importances[feature])
        aa=aa+importances[feature]
    print(aa)
    print('---------------')
    aa=0
    for feature in feature_names2:
        print(X1.columns[feature])
        print(importances[feature])
        aa=aa+importances[feature]
    print(aa)
    print('---------------')
    aa=0
    for feature in feature_names3:
        print(X1.columns[feature])
        print(importances[feature])
        aa=aa+importances[feature]
    print(aa)
    print('---------------')
    aa=0
    for feature in feature_names4:
        print(X1.columns[feature])
        print(importances[feature])
        aa=aa+importances[feature]
    print(aa)
#data is the file name that integrates all the data.
importance_rf(data)    


def importance_XGB(data):
    aa=0
    X1 = data.drop(['ESG','code','year'], axis=1)
    y = data['ESG']
    X = X1.dropna()
    y = y[X.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X=pd.DataFrame(X)  
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=2)
    xgb_model = xgb.XGBRegressor(n_estimators=150, random_state=0,reg_alpha=5, reg_lambda=5)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"样本外均方误差: {mse}")
    y_pred_train = xgb_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print(f"样本内拟合优度（训练集）: {r2_train}")
    r2_test = r2_score(y_test, y_pred)
    print(f"样本外拟合优度（测试集）: {r2_test}")
    importances = xgb_model.feature_importances_
    feature_names = [6,7,8,9,10,11,12,13,14,15]#O
    feature_names2 = [15,16,17,18,19,20,21,22,23,24]#D
    feature_names3 = [25,26,27,28,29,30,31,32,33,34]#P
    feature_names4 = [35,36,37,38,39,40,41,42,43,44]#G
    for feature in feature_names:
        print(X1.columns[feature])
        print(importances[feature])
        aa=aa+importances[feature]
    print(aa)
    aa=0
    print('---------------')    
    for feature in feature_names2:
        print(X1.columns[feature])
        print(importances[feature])
        aa=aa+importances[feature]
    print(aa)
    print('---------------')
    aa=0
    for feature in feature_names3:
        print(X1.columns[feature])
        print(importances[feature])
        aa=aa+importances[feature]
    print(aa)
    print('---------------')
    aa=0
    for feature in feature_names4:
        print(X1.columns[feature])
        print(importances[feature])
        aa=aa+importances[feature]
    print(aa)
    for feature in feature_names:
        print(X1.columns[feature])
        print(importances[feature])
        aa=aa+importances[feature]
    print(aa)
importance_XGB(data)    
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



















