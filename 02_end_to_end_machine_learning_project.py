from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("ziazhte/datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("ziazhte").mkdir(parents=True,exist_ok=True)
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url  = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("ziazhte/datasets/housing/housing.csv"))

housing = load_housing_data()

# The info() method is useful to get a quick description of the data, in particular the total
# number of rows, each attribute's type, and the number of nun-null values:
print(housing.info())

#You can find out what categories exist and how manu districts belong to each category
# by using the below command
print(housing["ocean_proximity"].value_counts())

# The describe method shows a summary of the numerical attributes
print(housing.describe())

# extra code – code to save the figures as high-res PNGs for the book

IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import matplotlib.pyplot as plt

plt.rc('font', size=14, family='serif')# Sets a global font size and family for all titles and labels
plt.rc('axes', labelsize=14, titlesize=14) # Sets the size of the labes like median income etc
plt.rc('legend', fontsize=14)# legend is the small box on the topleft corner showing diff colors
# for test and train as in the case of daneil bourke pytorch
plt.rc('xtick', labelsize=10)#Changes the size and color of the tick marks
plt.rc('ytick', labelsize=10)

#housing.hist(bins=50, figsize=(12,8))
# save_fig("attribute_histogram_plots")  # extra code
# plt.show()

# Creating the test set randomly

import numpy as np

# def shuffle_and_split_data(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

#ANCHOR - Implementing a hash_id

from zlib import crc32

# def is_id_in_test(identifier, test_ratio):
#     return crc32(np.int64(identifier)) < test_ratio * 2 **32

# def split_data_with_id_hash(data, test_ratio, id_column):
#     ids= data[id_column]
#     in_test_set = ids.apply(lambda id_: is_id_in_test(id_,test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]

# housing_with_id = housing.reset_index() # add an index column
# #print(housing_with_id.head())
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
# #print(len(test_set))

# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
# print(housing_with_id.head())
# print(len(test_set))

#ANCHOR - Train_test_split using SCIKIT-LEARN

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],#np.inf is a place ho
                               #holder for missing data
                               labels=[1,2,3,4,5])
# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0,grid=True)
# plt.xlabel("income category")
# plt.ylabel("Number of districts")
# plt.legend() # it gives the box in the top right corner
# save_fig("Income category")  # extra code
# plt.show()

#ANCHOR - Splitting stratified data in multiple train_test splits of same data

'''
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size= 0.2, random_state=42)
strat_splits = []
i = 1
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n,strat_test_set_n])
    i+=1
    #print(i)
    #print(len(strat_splits))
strat_train_set, strat_test_n = strat_splits[0]'''

#ANCHOR - Splitting stratified data in single train_test splits of same data since
# our dataset is quite small

strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2,
                                                   stratify=housing["income_cat"],
                                                      random_state=42)
# Now we won't use income_cat column we can drop it reverting data back to its original state
#print(strat_train_set.head())
#strat_train_set = strat_train_set.drop('income_cat',axis=1)# Instead of using the for
#loop we can directly use the above code to drop a column 

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace = True)

#print(strat_train_set.head())

housing = strat_train_set.copy()

#ANCHOR - Creating a scatter plot of all the districts tovisualize the data

# housing.plot(kind="scatter", x="longitude", y='latitude', grid=True,
#              s=housing["population"]/100, label="population",
#              c="median_house_value", cmap="jet",colorbar=True,
#              legend=True, sharex=False,figsize=(10,7))
# save_fig("Geographical scatterplot")
# plt.show()

#ANCHOR - Compute STANDARD CORRELATION COEFFICIENT

corr_matrix = housing.corr(numeric_only=True)# numeric_only is imp since we have datatype object
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#ANCHOR - Scatter matrix plot for every numerical attribute
"""
from pandas.plotting import scatter_matrix

attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))
save_fig("scatter matrix plots")
plt.show()"""

# housing.plot(kind="scatter", x="median_income",y="median_house_value",alpha=.1,grid=True)
# save_fig("med inc vs med hous val")
# plt.show()

# ANCHOR - Experiment with Attribute Combinations
# we will look for the combinations of the attributes which are less responsive 
# to median_house_value

housing["rooms_per_house"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["people_per_house"] = housing["population"]/housing["households"]

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#ANCHOR - Let''s prepare the Data for Machine Learning algorithms
# transform, separate the predictors and labels data,

# Separate the predictors and labels data

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#ANCHOR - Clean the data
# fixiing the missing features

# To get rid of the corresponding districts
# housing.dropna(subset=["total_bedrooms"],inplace=True)

# To get rid or the whole attribute
# housing.drop("total_bedrooms", axis=1)

# Set the missing values to some value(zero,mean,median etc) this is called IMPUTATION  
median  = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

#ANCHOR - sklearn Simple Imputer... Instead of using above we can use sklearn's
# SimpleImputer which will do on all train,test,validate and the new data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

# simple imputer can only work on numerical data so remove proximity column
housing_num = housing.select_dtypes(include=[np.number])
print(housing_num.info())
'''# housing_num = housing.drop("ocean_proximity",axis=1)
# print(housing_num.info())'''
# any of the above 2 codes can be used
imputer.fit(housing_num)# fit is training function
# Now you can use this "trained" imputer to transform the training set by
# replacing missing values with the learned medians
X = imputer.transform(housing_num)#transform is a kind of predicting function

print(X)
# The above code gives a numpy array to get back the data frame we need to do this
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
print(housing_tr.head())

#ANCHOR - Handling text and categorical attributes
# So we need to convert these ocean_proximity from text to numbers
from sklearn.preprocessing import OrdinalEncoder

housing_cat = housing[["ocean_proximity"]]# here double square bracket is necessary for sklearn
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

#ANCHOR - Hot encoder is good than ordinal encoder '

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat).toarray()
# toarray is used to convert from sparse matrix to numpy array
# Alternately you can set sparse=False when creating the OneHotEncoder,
# in which case transform() method will return a regular Numpy array directly
# print(housing_cat_1hot)
df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN","ISLAND"]})
pd.get_dummies(df_test_unknown)
cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)
print(cat_encoder.feature_names_in_)
print(cat_encoder.get_feature_names_out())
# df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
#                          columns=cat_encoder.get_feature_names_out(),
#                          index= df_test_unknown.index)
# print(df_output)

#ANCHOR - Scaling the attributes
# 1. MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# 2. StandaraScaler
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# 3. scaling Using RBF kernel
from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]],[[35]],gamma=0.1)

#ANCHOR - Scaling the output labels
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels  = target_scaler.fit_transform(housing_labels.to_frame())# to frame
#converts a  1D pandas Series into a 2D column vector
model = LinearRegression()
model.fit(housing[["median_income"]],scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]#pretend this as new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

#ANCHOR - Transformed Target Regressor
# This method does the above work in one line
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),transformer=StandardScaler())
model.fit(housing[["median_income"]],housing_labels)
predictions = model.predict(some_new_data)

from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[['population']])

#Code to create a transformer that computes the same gaussian RBF similarity as above

rbf_transformer = FunctionTransformer(rbf_kernel,kw_args=dict(Y=[[35]],gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords],gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude","longitude"]])

""" Function Transformer are also useful to combine features.
ratio_transformer  = FunctionTransformer(lambda X:X[:,[0]]/X[:,[1]])
ratio_transformer.transform(np.array([[1.,2.],[3.,4.]]))"""

#ANCHOR - Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):# y is required eventhough we dont use it
        X= check_array(X) # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)#The trailing underscore (_) is a strict scikit-learn convention.
        #axis=0 specifies that the operation should run down the rows for each column.
        # If you have a matrix of shape (100, 4)—meaning 100 samples and 4 features—axis=0
        #  will compress the 100 rows into 1 average value per column, resulting in
        #  an array of 4 mean values.
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1] # every estimator stores this in fit()
        return self # always return self

    def transform(self,X):
        check_is_fitted(self) # looks for learned attributes( with trailing_)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]#This line of code checks if the 
        #number of columns in your new data X exactly matches the number of features
        #  your model or transformer was originally trained on (n_features_in_).
        if self.with_mean:
            X = X - self.mean_
        return X/self.scale_

#ANCHOR - Custom transformer for ClusterSimilarity
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self,n_clusters=10, gamma=1.0, random_state = None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X,y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
    
    def transform(self,X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster{i} similarity" for i in range(self.n_clusters)] 

cluster_simil = ClusterSimilarity(n_clusters=10,gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude","longitude"]],
                                           sample_weight=housing_labels)
print(similarities[:2].round(2))

'''Extra code for plotting
housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (usᴅ)"})
housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                     s=housing_renamed["Population"] / 100, label="Population",
                     c="Max cluster similarity",
                     cmap="jet", colorbar=True,
                     legend=True, sharex=False, figsize=(10, 7))
plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
         cluster_simil.kmeans_.cluster_centers_[:, 0],
         linestyle="", color="black", marker="X", markersize=20,
         label="Cluster centers")
plt.legend(loc="upper right")
save_fig("district_cluster_plot")
plt.show()'''

#ANCHOR - Transformation pipelines
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
     ("standardize",StandardScaler()),
])
# if we dont want to give names to transformers
from sklearn.pipeline import make_pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy="median"),StandardScaler())
housing_num_prepared = num_pipeline.fit_transform(housing_num)

# if you want to recover the dataframe, you can do this
df_housing_num_prepared = pd.DataFrame(housing_num_prepared,
                                       columns=num_pipeline.get_feature_names_out(),
                                       index=housing_num.index)
# Now for categorical pipeline
cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))

# sofar we have handled num and cat cols separately now we will do by combining both
from sklearn.compose import ColumnTransformer

num_attributes = ["longitude","latitude","house_median_age","total_rooms",
                  "total_bedrooms","population","households","median_income"]
cat_attributes = ["ocean_proximity"]

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
])

# If the no of features is large and we don't to specify the names of the numerical 
# col names and cat col names separately we can use this they will be automatically 
# named to pipeline1, pipeline2 instead of num and cat

from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)

def column_ratio(X):
    return X[:,[0]]/X[:,[1]]
#X[:, [0]]: Selects all rows for column 0. The brackets [[0]] ensure the output 
# remains a 2D column vector.
# X[:, [1]]: Selects all rows for column 1, keeping it as a 2D column vector.
# /: Divides the elements of column 0 by the elements of column 1.

def ratio_name(function_transformer, feature_names_in):
    #The code snippet def ratio_name(function_transformer, feature_names) 
    # is a helper function used in scikit-learn to dynamically name new columns 
    # generated by a FunctionTransformer.
    return ["ratio"]
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)
cluster_simil = ClusterSimilarity(n_clusters=10,gamma=1.,random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
    ("bedrooms",ratio_pipeline(),["total_bedrooms","total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ("geo",cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],
remainder=default_num_pipeline) #This applies your numerical pipeline to any remaining 
#columns (housing_median_age) that were not explicitly listed in the transformer steps.

housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)
print(preprocessing.get_feature_names_out())

#ANCHOR - Select and Train the model

#ANCHOR - Liner Regression- let's start by simple Model
from sklearn.linear_model import LinearRegression

lin_reg= make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions  = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))# -2 = rounded to the nearest hundred 12345 = 12300
print(housing_labels.iloc[:5].values)

from sklearn.metrics import root_mean_squared_error
lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print(lin_rmse)

#ANCHOR - DecisionTreeRegressor
# since we did'nt got predictions on the training set the model is underfitting 
# the data so we need some more powerful model or additional features 

from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print(tree_rmse)

#ANCHOR -  Cross Validation
# k-folds cross validation feature
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg,housing,housing_labels,
                              scoring="neg_root_mean_squared_error",cv=10)
# we need put - sign above to make it positive
print(pd.Series(tree_rmses).describe())

# implementing to see how linear regression model works using cross vali score
# tree_rmses_lin = -cross_val_score(lin_reg, housing,housing_labels,
#                                   scoring="neg_root_mean_squared_error",cv=10)
# print(pd.Series(tree_rmses_lin).describe())

#ANCHOR -  Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# forest_reg = make_pipeline(preprocessing,RandomForestRegressor(random_state=42))
# forest_rmses = -cross_val_score(forest_reg,housing, housing_labels,
#                                 scoring="neg_root_mean_squared_error",cv=10)
# print(pd.Series(forest_rmses).describe())

#ANCHOR - Hypermeter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
'''param_grid = [
    {"preprocessing__geo__n_clusters": [5,8,10],
     "random_forest__max_features":[4,6,8]},
     {"preprocessing__geo__n_clusters":[10,15],
      "random_forest__max_features":[6,8,10]}]

grid_search = GridSearchCV(full_pipeline,param_grid,cv=3,
                           scoring="neg_root_mean_squared_error")
grid_search.fit(housing, housing_labels)
print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False,inplace=True)
print(cv_res.head())'''

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3,high=30),
                  'random_forest__max_features': randint(low=2,high=20)}
rnd_search = RandomizedSearchCV(
    full_pipeline,param_distributions=param_distribs,n_iter=10,cv=3,
    scoring='neg_root_mean_squared_error',random_state=42)
rnd_search.fit(housing, housing_labels)

#ANCHOR - Taking the best predictor in the RandomForestRegressor
final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
print(sorted(zip(feature_importances,
                 final_model["preprocessing"].get_feature_names_out()),reverse=True))

#ANCHOR - Evaluate your system on the Test set

X_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test,final_predictions)
print(final_rmse)

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) -1,
loc=squared_errors.mean(),scale = stats.sem(squared_errors))))