import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


def grid_search(data, labels, test_size=0.25, multi_class=False, normalizer_function=MinMaxScaler):

    # Set the parameters by cross-validation
    gamma = [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    c_weight = [0.01, 0.1, 1, 5, 10, 20, 50, 100, 200, 500, 1000]
    deg = [3, 6, 9]
    class_weight = [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}, {0: 1, 1: 3}, {0: 3, 1: 1}, {0: 1, 1: 4}, {0: 4, 1: 1}]

    hyper_parameters = [
        {'kernel': ['rbf'], 'gamma': gamma, 'C': c_weight, 'class_weight': class_weight},
        {'kernel': ['linear'], 'C': c_weight, 'class_weight': class_weight},
        {'kernel': ['poly'], 'degree': deg, 'gamma': gamma, 'C': c_weight, 'class_weight': class_weight}
    ]
    if multi_class:
        for el in hyper_parameters:
            el['decision_function_shape'] =  ['ovo']

    # Split the data set in two equal parts
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=None)

    normalizer = normalizer_function()
    normalizer.fit(x_train)
    x_train = normalizer.transform(x_train)
    x_test = normalizer.transform(x_test)

    for score in ['accuracy']:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), hyper_parameters, cv=5, scoring=score, n_jobs=6, verbose=5)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))
        print()

    return clf, [x_train, x_test, y_train, y_test]


def compute_confusion(classifier, data_list):
    c_acc = None
    names = ['train', 'test']
    for i, n in enumerate(names):
        yp = classifier.predict(data_list[i])
        print("Results for data: {}".format(n))
        c_acc = np.mean(data_list[i + 2] == yp)
        print(c_acc)
        print(confusion_matrix(data_list[i + 2], yp))

    return c_acc
