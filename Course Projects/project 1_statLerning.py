from sklearn import datasets, discriminant_analysis, model_selection, neighbors, metrics, linear_model
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import timeit

breast = datasets.load_breast_cancer()
iris =datasets.load_iris()
digit = datasets.load_digits()

iris_data = iris['data']
iris_targets = iris['target']

# split the model into train and test set
iris_data_train, iris_data_test, iris_targets_train , iris_targets_test = \
    model_selection.train_test_split(iris_data, iris_targets, test_size=0.25)

# K-nearest neighbor classifier
knn_iris = neighbors.KNeighborsClassifier(n_neighbors=13)
knn_iris.fit(iris_data_train, iris_targets_train)

iris_target_pred_knn = knn_iris.predict(iris_data_test)
# Another way to get accuracy explicitly is: np.mean(y == y_pred)
print("KNN Accuracy (Iris Dataset):",metrics.accuracy_score(iris_targets_test, iris_target_pred_knn))

# K-fold cross validation
iris_kfold = model_selection.KFold(n_splits=4, shuffle=True)
cv_score = model_selection.cross_val_score(knn_iris, X=iris_data, y=iris_targets, cv=iris_kfold)
print("Cross-validation score is %s" % cv_score, "Mean CV is %s" % np.mean(cv_score))

# Compare the performance of kNN for different k:
k_accuracy_scores = np.zeros((49, 400))
# for k in range(2, 51):
#     for rep in range(1, 400):
#         iris_data_train, iris_data_test, iris_targets_train, iris_targets_test = \
#             model_selection.train_test_split(iris_data, iris_targets, test_size=0.25)
#         knn_test_iris = neighbors.KNeighborsClassifier(n_neighbors=k)
#         knn_test_iris.fit(iris_data_train, iris_targets_train)
#         iris_ypred_test = knn_test_iris.predict(iris_data_test)
#         k_accuracy_scores[k-2, rep] = metrics.accuracy_score(iris_targets_test, iris_ypred_test)
#
# #k_prediction_times = np.divide(k_prediction_times, max(k_prediction_times))
# x_axis = np.arange(2, 51)
# plt.plot(x_axis, np.mean(k_accuracy_scores, axis = 1))
# plt.xlabel('k')
# plt.ylabel('Accuracy Score')
# plt.title('Accuracy score of kNN for different values of k averaged over 400 repeats')
# plt.show()

# X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap = colors.ListedColormap(['cadetblue', 'firebrick', 'gold'])
cmap_scat = colors.ListedColormap(['blue', 'red', 'yellow'])

for k in [2, 5]:
    knn_iris = neighbors.KNeighborsClassifier(n_neighbors = k)
    knn_iris.fit(X, y)
    y_pred = knn_iris.predict(X)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    Z = knn_iris.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap)

# Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_scat, edgecolor='k', s=20)
    plt.title("kNN Classification of Iris Dataset, k = %s Accuracy without split : %0.4f" % (k, np.mean(y == y_pred)))
    plt.axis('tight')

plt.show()
#
# # Nearest Centroid Classifier
# nc_iris = neighbors.NearestCentroid()
# nc_iris.fit(iris_data_train, iris_targets_train)
#
# iris_targets_pred_nc = nc_iris.predict(iris_data_test)
# print("Nearest Centroid Accuracy (Iris Dataset):", metrics.accuracy_score(iris_targets_test, iris_targets_pred_nc))
#
# nc_iris.fit(X, y)
# y_pred = nc_iris.predict(X)
#
# Z = nc_iris.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_scat, edgecolor='k', s=20)
# plt.title("Nearest Centroid Classification of Iris Dataset \
#            Accuracy without test-train split : %0.4f  \
#            Accuracy with test-train split : %0.4f" %
#           (metrics.accuracy_score(y, y_pred), metrics.accuracy_score(iris_targets_test, iris_targets_pred_nc)))
# plt.axis('tight')
# plt.show()
#
# # QDA classifier
# qda_iris = discriminant_analysis.QuadraticDiscriminantAnalysis()
# qda_iris.fit(iris_data_train, iris_targets_train)
# iris_targets_pred_qda = qda_iris.predict(iris_data_test)
#
# qda_iris.fit(X, y)
# y_pred = qda_iris.predict(X)
#
# Z = qda_iris.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap)
#
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_scat, edgecolor='k', s=20)
# plt.title("QDA Classification of Iris Dataset \
#           Accuracy without split: %0.4f \
#           Accuracy with train-test split: %0.4f" % (metrics.accuracy_score(y, y_pred), metrics.accuracy_score(iris_targets_test, iris_targets_pred_qda)))
# plt.axis('tight')
# plt.show()
#
# #Logistic Regression
# c_logreg_values = np.arange(0.01, 2.01, 0.01)
# intercept_scaling_values = np.arange(0.01, 2.01, 0.01)
# logreg_liblinear_acc= []
# logreg_acc_lbfgs_multi = []
#
# for c in c_logreg_values:
#     logreg_iris = linear_model.LogisticRegression(C=c , multi_class='multinomial', max_iter=1000)
#     logreg_iris.fit(iris_data_train, iris_targets_train)
#     iris_ypred_test = logreg_iris.predict(iris_data_test)
#     logreg_acc_lbfgs_multi.append(metrics.accuracy_score(iris_targets_test, iris_ypred_test))
#
# for c in c_logreg_values:
#     logreg_iris = linear_model.LogisticRegression(C=c, solver='liblinear')
#     logreg_iris.fit(iris_data_train, iris_targets_train)
#     iris_ypred_test = logreg_iris.predict(iris_data_test)
#     logreg_liblinear_acc.append(metrics.accuracy_score(iris_targets_test, iris_ypred_test))
#
# x_axis = c_logreg_values
# plt.plot(x_axis, logreg_acc_lbfgs_multi, x_axis, logreg_liblinear_acc)
# plt.title('Logistic Regression Accuracy Scores for different parameters and solvers')
# plt.legend(('lbgfs with multinomial multiclass', 'liblinear'))
# plt.xlabel('C')
# plt.ylabel('Accuracy Score')
# plt.axis('tight')
# plt.show()

# accuracy_matrix = np.zeros((len(intercept_scaling_values), (len(intercept_scaling_values))))
# for ii in range(np.shape(accuracy_matrix)[0]):
#     for jj in range(np.shape(accuracy_matrix)[1]):
#         logreg_iris = linear_model.LogisticRegression(C=ii, solver='liblinear', intercept_scaling=jj)
#         logreg_iris.fit(iris_data_train, iris_targets_train)
#         iris_ypred_test = logreg_iris.predict(iris_data_test)
#         accuracy_matrix[ii, jj] = metrics.accuracy_score(iris_targets_test, iris_ypred_test)
#
# index = np.where(accuracy_matrix == set(max(accuracy_matrix)))
# print(index)

# Breast Data
# breast_data = breast['data']
# breast_targets = breast['target']
#
# breast_data_train, breast_data_test, breast_targets_train , breast_targets_test = \
#     model_selection.train_test_split(breast_data, breast_targets, test_size=0.25)
#
# knn_breast = neighbors.KNeighborsClassifier(n_neighbors=2)
# knn_breast.fit(breast_data_train, breast_targets_train)
#
# breast_target_pred = knn_breast.predict(breast_data_test)
# print("KNN Accuracy (Breast Dataset):",metrics.accuracy_score(breast_targets_test, breast_target_pred))
#
#
# nc_breast = neighbors.NearestCentroid()
# nc_breast.fit(breast_data_train, breast_targets_train)
#
# breast_targets_pred_nc = nc_breast.predict(breast_data_test)
# print("Nearest Centroid Accuracy (Breast Dataset):",metrics.accuracy_score(breast_targets_test, breast_targets_pred_nc))
#
# qda_breast = discriminant_analysis.QuadraticDiscriminantAnalysis()
# qda_breast.fit(breast_data_train, breast_targets_train)
# breast_targets_pred_qda = qda_breast.predict(breast_data_test)
# print("QDA Accuracy (Breast Dataset):",metrics.accuracy_score(breast_targets_test, breast_targets_pred_qda))
#
# X = breast.data[:, :2]
# y = breast.target
#
# h = .05  # step size in the mesh
#
# # Create color maps
# cmap = colors.ListedColormap(['cadetblue', 'firebrick', 'gold'])
# cmap_scat = colors.ListedColormap(['blue', 'red', 'yellow'])
#
# qda_breast.fit(X, y)
# y_pred = qda_breast.predict(X)
# print(np.mean(y == y_pred))
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# Z = qda_breast.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_scat, edgecolor='k', s=20)
# plt.title("3-Class classification")
# plt.axis('tight')
# plt.show()

#Digit Data
digit_data = digit['data']
digit_targets = digit['target']

digit_data_train, digit_data_test, digit_targets_train , digit_targets_test = \
    model_selection.train_test_split(digit_data, digit_targets, test_size=0.25)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(digit_data_train, digit_targets_train)

digit_target_pred = knn.predict(digit_data_test)
print("KNN Accuracy (Digit Dataset):",metrics.accuracy_score(digit_targets_test, digit_target_pred))

# k_accuracy_scores = np.zeros((49, 100))
# for k in range(2, 51):
#     for rep in range(1, 100):
#         digit_data_train, digit_data_test, digit_targets_train, digit_targets_test = \
#             model_selection.train_test_split(digit_data, digit_targets, test_size=0.25)
#         knn_test_digit = neighbors.KNeighborsClassifier(n_neighbors=k)
#         knn_test_digit.fit(digit_data_train, digit_targets_train)
#         digit_ypred_test = knn_test_digit.predict(digit_data_test)
#         k_accuracy_scores[k-2, rep] = metrics.accuracy_score(digit_targets_test, digit_ypred_test)
#
# #k_prediction_times = np.divide(k_prediction_times, max(k_prediction_times))
# x_axis = np.arange(2, 51)
# plt.plot(x_axis, np.mean(k_accuracy_scores, axis = 1))
# plt.xlabel('k')
# plt.ylabel('Accuracy Score')
# plt.title('Accuracy score of kNN for different values of k for the digits dataset')
# plt.show()

#Logistic Regression
# c_logreg_values = np.arange(0.01, 2.01, 0.01)
# intercept_scaling_values = np.arange(0.01, 2.01, 0.01)
# logreg_liblinear_acc= []
# logreg_acc_lbfgs_multi = []
#
# for c in c_logreg_values:
#     logreg_digit = linear_model.LogisticRegression(C=c , multi_class='multinomial', max_iter=5000)
#     logreg_digit.fit(digit_data_train, digit_targets_train)
#     digit_ypred_test = logreg_digit.predict(digit_data_test)
#     logreg_acc_lbfgs_multi.append(metrics.accuracy_score(digit_targets_test, digit_ypred_test))
#
# for c in c_logreg_values:
#     logreg_digit = linear_model.LogisticRegression(C=c, solver='liblinear')
#     logreg_digit.fit(digit_data_train, digit_targets_train)
#     digit_ypred_test = logreg_digit.predict(digit_data_test)
#     logreg_liblinear_acc.append(metrics.accuracy_score(digit_targets_test, digit_ypred_test))
#
# x_axis = c_logreg_values
# plt.plot(x_axis, logreg_acc_lbfgs_multi, x_axis, logreg_liblinear_acc)
# plt.title('Logistic Regression Accuracy Scores for different parameters and solvers')
# plt.legend(('lbgfs with multinomial multiclass', 'liblinear'))
# plt.xlabel('C')
# plt.ylabel('Accuracy Score')
# plt.axis('tight')
# plt.show()

nc_digit = neighbors.NearestCentroid()
nc_digit.fit(digit_data_train, digit_targets_train)

digit_targets_pred_nc = nc_digit.predict(digit_data_test)
print("Nearest Centroid Accuracy (Digit Dataset):",metrics.accuracy_score(digit_targets_test, digit_targets_pred_nc))


qda_digit = discriminant_analysis.QuadraticDiscriminantAnalysis()
qda_digit.fit(digit_data_train, digit_targets_train)
digit_targets_pred_qda = qda_digit.predict(digit_data_test)
print("QDA Accuracy (Digit Dataset):",metrics.accuracy_score(digit_targets_test, digit_targets_pred_qda))

# iris_data_split = np.reshape(iris_data, [10, len(iris_data)/10])
# iris_targets_split = np.reshape(iris_targets [10, len(iris_data)/10])

#K-fold cross validation