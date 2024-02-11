import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import warnings
warnings.simplefilter(action="ignore")

#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test2 = test["Adj Close"]
test = test.drop("Adj Close", axis=1)
#target = df["Adj Close"]
df = pd.concat([train, test], ignore_index=False).reset_index()
df = df.drop("index", axis=1)
#print(train)
#print(test)
print(df)
df["Date"] = pd.to_datetime(df["Date"])
#print(df.dtypes)
#print(df.info())
#print(df.isnull().sum())

#######################################
# ANALYSİS OF CATEGORICAL VARIABLES
#######################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, cat_but_car, num_cols


cat_cols, cat_but_car, num_cols = grab_col_names(df)
#print(cat_cols)
#print(cat_but_car)
#print(num_cols)


###################################
# ANALYSIS OF NUMERİCAL VARIABLES
###################################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


#for col in num_cols:
#    num_summary(df, col, True)


################################
# ANALYSIS OF TARGET VARIABLE
################################
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


#for col in num_cols:
#    target_summary_with_cat(df, "Adj Close", col)


##############################
# ANALYSIS OF CORRELATION
##############################

corr = df[num_cols].corr()
#print(corr)

#sns.set(rc={"figure.figsize": (12, 12)})
#sns.heatmap(corr, cmap="RdBu")
#plt.show()


######################################
# OUTLIER ANALYSIS
######################################
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


#for col in num_cols:
#    print(col, check_outlier(df, col))


#########################
# BASE MODEL BUILDING
#########################
train_df = df[df["Adj Close"].notnull()]
test_df = df[df["Adj Close"].isnull()]

y = train_df["Adj Close"] # np.log1p(df["SalePrice"])
X = train_df.drop(["Date", "Adj Close"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

linear_model = LinearRegression().fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
predictions = linear_model.predict(test_df.drop(["Date", "Adj Close"], axis=1))
dictionary = {"Date": test_df.index, "Adj Close": predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("tsla1.csv", index=False)


df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df["day"] = df["Date"].dt.day.astype("int")
df["month"] = df["Date"].dt.month.astype("int")
df["year"] = df["Date"].dt.year.astype("int")
#####################################################################################################
# Çeyrek Sonu Ayları ve Fiyatlar:
# Çeyrek sonu aylarında hisse senedi fiyatları, çeyrek sonu olmayan aylara göre daha yüksektir.
# Bu, şirketlerin çeyrek dönem sonuçlarını açıkladığı dönemlerde gerçekleşebilir. Yatırımcılar,
# bu dönemlerde şirket performansını değerlendirir ve bu da hisse senedi fiyatlarını etkiler.
#
# Çeyrek Sonu Ayları ve İşlem Hacmi:
# Çeyrek sonu aylarında işlem hacmi daha düşüktür. Bu, yatırımcıların çeyrek sonu dönemlerinde
# daha temkinli davrandığını veya daha az işlem yaptığını gösterebilir. Çünkü çeyrek sonu
# dönemlerinde şirketlerin finansal raporları ve sonuçları açıklanır ve bu dönemlerde
# belirsizlik artabilir.
#####################################################################################################
df["is_quarter_end"] = np.where(df["month"] % 3 == 0, 1, 0)

#data_grouped = df.groupby("year").mean()
#plt.subplots(figsize=(20, 10))
#for i, col in enumerate(["Open", "High", "Low", "Close"]):
#    plt.subplot(2, 2, i + 1)
#    data_grouped[col].plot.bar()
#plt.show()

df["open-close"] = df["Open"] - df["Close"]
df["low-high"] = df["Low"] - df["High"]
###########################################################################################
# her günün kapanış fiyatını bir sonraki günün kapanış fiyatıyla karşılaştırır ve eğer
# bir sonraki günün kapanış fiyatı daha yüksekse 1, aksi takdirde 0 olarak işaretlenir.
###########################################################################################
df["target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
features = df[["open-close", "low-high", "is_quarter_end"]]
target = df["target"]

##############################################################################
# pasta grafiği kullanarak hedefin dengeli olup olmadığını kontrol edelir.
##############################################################################
#plt.pie(df["target"].value_counts().values, labels=[0, 1], autopct="%1.1f%%")
#plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

train_df = df[df["Adj Close"].notnull()]
test_df = df[df["Adj Close"].isnull()]

y2 = train_df["Adj Close"] # np.log1p(df["SalePrice"])
X2 = train_df.drop(["Date", "Adj Close"], axis=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.20, random_state=17)

linear_model = LinearRegression().fit(X_train2, y_train2)
y_pred2 = linear_model.predict(X_test2)
print(y_pred2)

