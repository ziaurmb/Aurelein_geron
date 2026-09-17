import numpy as np

np.random.seed(42)# 42 to make this code example reproducible
m = 100 # no of instances
X = 2*np.random.rand(m,1) # column vector
y = 4 + 3 * X + np.random.randn(m,1) #column vector

import matplotlib.pyplot as plt

plt.rc("font", size = 14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

from pathlib import Path
IMAGES_PATH = Path() / "images" / "04_images_training_lin_mod"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id,tight_layout=True, fig_extension ="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import matplotlib.pyplot as plt

'''
plt.figure(figsize=(6,4))
plt.plot(X,y,"b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.grid()
save_fig("generated_data_plot")
plt.show()'''

from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X) # add x0 = 1 to each instance, this 
#print(X,X_b)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
#LINK - theta_best_1 = np.linalg.inv(X.T @ X) @ X.T @ y
# print(theta_best)
# print(theta_best_1)

# Now make prediction susing theta hat
X_new = np.array([[0],[1],[2]])
y_new = 4 + 3 * X_new + np.random.randn(3,1) # original output
# print(X_new[1])
X_new_b = add_dummy_feature(X_new)
y_predict = X_new_b @ theta_best # predicted output
# print(y_new)
# print(y_predict)
  # extra code – not needed, just formatting
'''
plt.figure(figsize=(6,4))
plt.plot(X_new, y_predict, "r-", label="Predictios")
plt.plot(X_new,y_predict,"g.",markersize=20)
plt.plot(X, y, "b.")
plt.xlabel("x_1")
plt.ylabel("y",rotation=0)
plt.axis([0,2,0,15])
plt.grid()
plt.legend(loc="upper left")
save_fig("linear_model_predictions_plot")
plt.show()'''

#ANCHOR - Now performing linearRegression using sklearn is straightforward

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.coef_,lin_reg.intercept_) # gives theta0 and theta1 values
print(lin_reg.predict(X_new))
eta = 0.1 # learning rate
n_epochs = 1000
m = len(X_b)# numbr of instances

np.random.seed(42)
theta = np.random.randn(2,1) # randomly initialized model parameters

for epoch in range(n_epochs):
    gradients = 2/m * X_b.T @ (X_b @ theta - y)
    theta  = theta - eta * gradients

print(theta)

n_epochs = 50
t0, t1 = 5,50 #learning schedule parameters

def learning_schedule(t):
    return t0/(t+t1)

np.random.seed(42)
theta = np.random.randn(2,1) # random initialization

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi) # for sgd do not divide by m
        eta = learning_schedule(epoch*m + iteration)
        theta = theta - eta * gradients
print(theta)

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5,penalty=None,eta0=0.01,
                       n_iter_no_change=100,random_state=42)
sgd_reg.fit(X, y.ravel()) # y.ravel() bec fit() expects 1D targets
print(sgd_reg.intercept_,sgd_reg.coef_)

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m,1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.rand(m,1)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01,1.0, 40),#Tells the function to
     # evaluate 40 different training set sizes
    #, starting from 1% of the data up to 100%., 
    cv=5,
    scoring="neg_root_mean_squared_error"
)
train_errors = -train_scores.mean(axis=1)
valid_errors  = -valid_scores.mean(axis=1)

'''
plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label= "train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=2, label="valid")
plt.axis([0,80,0,2.5])
plt.legend()
plt.grid()
plt.xlabel("Training set size")
plt.ylabel("RMSE")

save_fig("learning_curve_underfitting")
plt.show()
'''

from sklearn.pipeline import make_pipeline

polynomial_regression = make_pipeline(PolynomialFeatures(degree=10,include_bias=False),
                                                         LinearRegression())

train_sizes, train_scores, valid_scores  = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01,1.0,40), cv=5,
    scoring="neg_root_mean_squared_error"
)

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

'''
plt.plot(train_sizes , train_errors, "r-+", linewidth=2,label="Train")
plt.plot(train_sizes, valid_errors, "b-",linewidth=2,label="Valid")
plt.legend()
plt.grid()
plt.xlabel("Training set size")
plt.ylabel("RMSE",rotation=90)
plt.axis([0,80,0,2.5])
save_fig("learning_curve_overfitting")
plt.show()'''

#ANCHOR - REGULARIZATION
#ANCHOR - Ridge regression using sklearn

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.1,solver="cholesky")
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))

sgd_reg = SGDRegressor(penalty="l2",alpha=0.1/m, tol=None
                       ,max_iter=1000,eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))

#ANCHOR - Lasso Regression
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X,y)
print(X[0],y[0])
print(lasso_reg.predict([[1.5]]))

#ANCHOR - Elastic Net regression
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X,y)
print(elastic_net.predict([[1.5]]))

#ANCHOR - Basic implementation of early stopping
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

X_train, y_train, X_valid, y_valid = X[:80], y[:80].ravel(), X[80:], y[80:].ravel()

preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                                                 StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_pred = preprocessing.fit_transform(X_valid)
sgd_reg  = SGDRegressor(penalty=None,eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')#This code snippet is used to track and save the best-performing model

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep,y_train)
    y_valid_predict = sgd_reg.predict(X_valid_pred)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model  = deepcopy(sgd_reg)

#ANCHOR - Let's load IRIS data and make a cloassification model
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
print(iris.data.head(3))

# Now we'll split the data and train a logistic regression model on the training data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X= iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == "virginica"
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:,1] >= 0.5][0,0]
'''
plt.plot(X_new, y_proba[:,0],"b--",linewidth=2,
         label="Not Iris Verginica proba")
plt.plot(X_new, y_proba[:,1],"g-",linewidth=2,label = "Iris Verginica proba")
plt.plot([decision_boundary, decision_boundary],[0,1], "k:", linewidth=2,
         label="Decision Boundary")
plt.grid()
plt.legend(loc="center left")
plt.arrow(x=decision_boundary, y=0.08,dx=-0.3,dy =0,
          head_width=0.05, head_length=0.1,fc="b",ec="b")
plt.arrow(x=decision_boundary, y=0.92, dx=0.3, dy=0,
          head_width=0.05, head_length=0.1,fc="g", ec="g")
plt.plot(X_train[y_train == 0], y_train[y_train==0], "bs")
plt.plot(X_train[y_train == 1], y_train[y_train == 1], "g^")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.axis([0 , 3, -0.02, 1.02])
save_fig("logistic_regression_plot")
plt.show()'''

print(decision_boundary)
print(log_reg.predict([[1.7],[1.5]]))