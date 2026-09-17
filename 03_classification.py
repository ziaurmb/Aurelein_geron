# %load_ext cuml.accel
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target
# to see the shape
# print(X.shape)
# print(X[0])

from pathlib import Path
IMAGES_PATH = Path() / "03_images_classification" / "classifiction"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path   = IMAGES_PATH / F"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc('ytick', labelsize=10)

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
some_digit = X[0]
# plot_digit(some_digit)
# save_fig("some_digit_plot")
# plt.show()

X_train,X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')# this code gives the vector as [true,false,....]
# true for 5 and false for other 9 digits. Labelling will be changed from digits
# to only true and false.
y_test_5 = (y_test =='5')
#print(X_train.shape, y_train_5)

#ANCHOR - SGDClassifier -Let's train this on the whole data set 

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))

#ANCHOR -  Measuring the accuracy using cross-validation
from sklearn.model_selection import cross_val_score
#print(cross_val_score(sgd_clf,X_train, y_train_5, cv=3,scoring="accuracy"))

# Let's look at a dummy classifier that just classifies every single image in the
# most frequent class, which in this case is the negative class(i,e, non 5)
from sklearn.dummy import DummyClassifier

#ANCHOR - A "dummy classifier" is a baseline machine learning model used to 
# establish the absolute lowest acceptable performance metric. Classifying 
# every image as the "most frequent class" means the model completely ignores 
# the image content and simply guesses the single category that appears most often 
# in the training dataset.
dummy_clf = DummyClassifier()
# dummy_clf.fit(X_train, y_train_5)
# print(any(dummy_clf.predict(X_train))) # prints false: no 5s detected
# print(cross_val_score(dummy_clf,X_train, y_train_5,cv=3,scoring="accuracy"))

#ANCHOR - Implementing Cross-validation manually, this helps to get more
# control over the process

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

'''
skfolds = StratifiedKFold(n_splits=3)# add shuffle=True if the dataset is not already shuffled

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred==y_test_fold)
    print(n_correct/len(y_pred)) '''

#ANCHOR - Confusion Matrix- it's a way to evaluate the evaluate the performacne of the classifier
# Let's make prediction on the X_train using cross_val_pedict() using kfolds=3
# It will be trained for 3 times where each time it will be trained on 2 folds and one 
# will be used for cross validation, similarly it will be trained for 3 times
# giving predictions to the whole data.

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=5)

# import
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

# Let's consider we have got the perfect predictions
'''
y_train_perfect_pred  = y_train_5
print(confusion_matrix(y_train_5, y_train_perfect_pred))'''

#ANCHOR - Precision and Recall
# computing precision and recall using sklearn
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
'''
precision_score(y_train_5, y_train_pred)
# == 3530/(687+3530)
recall_score(y_train_5, y_train_pred)
# == 3530/(1891+3530)

# Now calculate F1 score
# F1 score is the harmonic mean of precision and recall
f1_score(y_train_5, y_train_pred)'''

#ANCHOR - Precision/Recall tradeoff using decision_function() in sklearn
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)#[2164.22030239]
threshold = 0 # setting threshold to 0 gives original predict() method
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)#[ True]
# If we increase the threshold value to 3000 then it will predict as false 
# it will predict True only if the threshold value is more then 3000,
threshold = 3000
y_some_digit_pred = (y_scores >= threshold)
print(y_some_digit_pred)#[ False]

#ANCHOR - How to decide which threshold to use?
# 1. use the cross_val_predict() to get the scores of all instances in the train set, 
# but this time specify that you want to return decision scores instead of predictions.
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, 
                             method="decision_function")
# 2. With these scores, use the precision_recall_curve() to compute precsion and 
# recall for all possible thresholds
from sklearn.metrics import precision_recall_curve
import matplotlib.patches as patches

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# 3. Use Matplotlib to plot precision and recall as functions of the threshold value(
# Let's show the threshold of 3000 in plot)
# Setting threshold to 3000
threshold = 3000
'''
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1],"g--", label = "Recall", linewidth=2)
plt.vlines(threshold,0,1.0,"k","dotted", label="threshold")
# extra code – this section just beautifies and saves Figure 3–5
idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
save_fig("precision_recall_vs_threshold_plot")

plt.show()

import matplotlib.patches as patches  # extra code – for the curved arrow

plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")

# extra code – just beautifies and saves Figure 3–6
plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
plt.plot([recalls[idx]], [precisions[idx]], "ko",
         label="Point at threshold 3,000")
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.79, 0.60), (0.61, 0.78),
    connectionstyle="arc3,rad=.2",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig("precision_vs_recall_plot")

plt.show()'''

# 4. Now calculate the estimated threshold
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
print(threshold_for_90_precision) # gives 3370.0194991439557

# To make prediction , instead of calling the precict( method try this)
y_train_pred_90 = (y_scores >= threshold_for_90_precision)

# let's chech these predictiosn
precision_score(y_train_5, y_train_pred_90)
recall_at_90_precision= recall_score(y_train_5, y_train_pred_90)

#ANCHOR - The ROC curve- 
# The Reciever Operating Characteristic is  another comm tool used with bin classifiers.

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
print(idx_for_threshold_at_90)

tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]
print(tpr_90,fpr_90)

'''
plt.plot(fpr, tpr, linewidth=2, label="ROC Curve")
plt.plot([0,1],[0,1],'k:',label= "Threshold for 90% precision")
# extra code – just beautifies and saves Figure 3–7
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.20, 0.89), (0.07, 0.70),
    connectionstyle="arc3,rad=.4",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)
save_fig("roc_curve_plot")
plt.show()'''

#ANCHOR - One way to compare classifiers is to measure the area under the curve(AUC)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

#ANCHOR - Let's now create a RandomForestClssifier, whose PR curve and F1 score we
# can compare to those of the SGDClassifier

# from sklearn.ensemble import RandomForestClassifier

# forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf,X_train, y_train_5, cv=3,
#                                     method="predict_proba")
# y_scores_forest = y_probas_forest[:,1]
# precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
#     y_train_5, y_scores_forest
# )

'''
plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")

# extra code – just beautifies and saves Figure 3–8
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig("pr_curve_comparison_plot")

plt.show()'''

# y_train_pred_forest = y_probas_forest[:,1] >= 0.5 # positive proba >= 50%
# print(f1_score(y_train_5,y_train_pred_forest))
# print(roc_auc_score(y_train_5, y_scores_forest))
# print(precision_score(y_train_5, y_train_pred_forest))
# recall_forest = recall_score(y_train_5, y_train_pred_forest)
# print(recall_forest)

from sklearn.svm import SVC

# svm_clf = SVC(random_state=42)
# svm_clf.fit(X_train[:2000],y_train[:2000])
# print(svm_clf.predict([some_digit]))

#ANCHOR - Check using decision_function() method who got the highest score
# some_digit_scores = svm_clf.decision_function([some_digit])
# print(some_digit_scores.round(2))# you can see '5' got the highest score of 9.3
# class_id  =  some_digit_scores.argmax()# this will predict the digit which has got
# # the max num. instead of using the predict method we can use this if we are using
# # decision_function() method
# print(class_id)

#ANCHOR - for binary classifiers like SVC and SGDClassifier the algorithms automatically
# assigns itslf to either OvO(OneVsOne) or OvR(OneVsRest) strategy. 
# In the above SVC model it used OvO strategy
# now we can force it to use OvR strategy

from sklearn.multiclass import OneVsRestClassifier
# ovr_clf = OneVsRestClassifier(SVC(random_state=42))
# ovr_clf.fit(X_train[:2000], y_train[:2000])
# print(ovr_clf.predict([some_digit]))

#ANCHOR - Traing an SGDClassifier on a multiclass dataset
sgd_clf = SGDClassifier(random_state=42, max_iter=3000)
#sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([some_digit]))
# print(sgd_clf.decision_function([some_digit]).round())

# Cross validating the model
# print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
print("sgd finished")
from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(
    StandardScaler(),
    SGDClassifier(random_state=42,max_iter=3000)
)
# score_zia = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy")
# print(score_zia)
#print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

from sklearn.metrics import ConfusionMatrixDisplay

# y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=3)
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
# plt.show()

# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, 
#                                         normalize="true", values_format=".0%")

#ANCHOR - MULTILABEL CLASSIFICATION
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 ==1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([some_digit]))

#ANCHOR - Evaluating multilable classifier one approach is to use f1 score

# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
# f1_score(y_multilabel, y_train_knn_pred,average="macro")

from sklearn.multioutput import ClassifierChain
chain_clf = ClassifierChain(SVC(),cv=3,random_state=42)
chain_clf.fit(X_train[:2000],y_multilabel[:2000])

#ANCHOR - MULTIOUTPUT CLASSIFICATION
# IT IS ALSO CALLED MULTIOUTPUT MULTICLASS CLASSIFICATION
# .

np.random.seed(42) # to make this code example reproducible
noise = np.random.randint(0,100,(len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0,100,(len(X_test),784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = y_test

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()

