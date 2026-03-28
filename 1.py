import pandas as pd
import numpy as np
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt
url="https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv"
df = pd.read_csv(url)
df.to_csv("ames_housiing.csv",index=False)
# print(df.head())
# print(df.describe())
# print(df.columns)
# print(df.shape)



#DATA CLEANING




pd.set_option('display.max_rows', None)
# print(df.isnull().sum().sort_values(ascending=False))
df = df.drop(columns=["Pool QC", "Alley", "Misc Feature", "Fence",'Unnamed: 0'])



none_cols = [
    "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
    "Mas Vnr Type", "Fireplace Qu"
]
df[none_cols] = df[none_cols].fillna("None")



# Numeric columns where NaN means 0 (no garage, no basement etc.)
zero_cols = [
    "Garage Yr Blt", "Garage Area", "Garage Cars",
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Bsmt Full Bath", "Bsmt Half Bath",
    "Mas Vnr Area"
]
df[zero_cols] = df[zero_cols].fillna(0)




df["Electrical"]   = df["Electrical"].fillna(df["Electrical"].mode()[0])


# print(df["Lot Frontage"])
df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())
df = pd.get_dummies(df)


# print(df.isnull().sum().sort_values(ascending=False))

# print(df.dtypes)





#EDA

from scipy import stats


results = []

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    if col != 'SalePrice':
        pearson_coef, p_value = stats.pearsonr(df[col], df['SalePrice'])
        results.append({
            'Feature'  : col,
            'Pearson'  : round(pearson_coef, 4),
            'P-Value'  : round(p_value, 4)
        })


results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Pearson', ascending=False)
# print(results_df.to_string())

# This handles numeric AND bool together correctly enough
correlation = df.corr()['SalePrice'].sort_values(ascending=False)
# print(correlation)




weak_features = correlation[abs(correlation) < 0.5].index.tolist()


if 'SalePrice' in weak_features:
    weak_features.remove('SalePrice')

df = df.drop(columns=weak_features)

# print(df.shape)
# print(df.columns.tolist())






import matplotlib.pyplot as plt
import seaborn as sns

# sns.boxplot(df['SalePrice'])
# plt.show()



Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# print("Lower bound:", lower)
# print("Upper bound:", upper)
print("Outliers:", df[(df['SalePrice'] < lower) | (df['SalePrice'] > upper)].shape[0])

df = df[(df['SalePrice'] >= lower) & (df['SalePrice'] <= upper)]
# print(df.shape)  # see how many rows remain





import matplotlib.pyplot as plt
import seaborn as sns

# sns.boxplot(df['SalePrice'])
# plt.show()


df.to_csv("final_data.csv",index=False)





#model development 



x_data=df.drop("SalePrice",axis=1)
y_data=df["SalePrice"]

# print("X shape:", x_data.shape)
# print("y shape:", y_data.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=1)

from sklearn.linear_model import LinearRegression
lre=LinearRegression()

lre.fit(x_train,y_train)
lre.fit(x_test,y_test)
print(lre.score(x_train,y_train))
print(lre.score(x_test,y_test))


# model evaluation and refinement

# corresponding r62 values are good but we may not have suffieient data thus cross validating
from sklearn.model_selection import cross_val_score,cross_val_predict
Rcross=cross_val_score(lre,x_data,y_data,cv=5)
print(Rcross)

yhat_train=lre.predict(x_train)
yhat_test=lre.predict(x_test)


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()


Title='Distribution  Plot of  Predicted Value Using train Data vs Data Distribution of Train Data'
DistributionPlot(y_train,yhat_train,"Actual Values (train)","Predicted Values (Train)",Title)


Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
lm=LinearRegression()
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
Rsqu_test = []

order = [1, 2, 3]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train)
    
    x_test_pr = pr.transform(x_test)    

    scaler_pr = StandardScaler()
    x_train_pr = scaler_pr.fit_transform(x_train_pr)
    x_test_pr = scaler_pr.transform(x_test_pr)
    
    lm.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lm.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.show()



# from plot it can be seen that polynomial feature is not a good idea



from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np

# Test many alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

results = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge,x_train_scaled, y_train, cv=5, scoring='r2')
    results.append({
        'Alpha'   : alpha,
        'Mean R2' : round(scores.mean(), 4),
        'Std Dev' : round(scores.std(), 4)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string())


best_alpha = results_df.loc[results_df['Mean R2'].idxmax(), 'Alpha']
print("Best Alpha:", best_alpha)

ridge_final = Ridge(alpha=best_alpha)
ridge_final.fit(x_train_scaled, y_train)

print("Train R2:", round(ridge_final.score(x_train_scaled, y_train), 4))
print("Test  R2:", round(ridge_final.score(x_test_scaled, y_test), 4))
import pickle

pickle.dump(ridge_final, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("Model saved!")

print(list(x_train.columns))






