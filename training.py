import sys

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def grid_search_notest(data, labels, multi_class=False, score='accuracy',
                normalizer_function=MinMaxScaler, classifier=SVC, hyper_parameters=None):

    if hyper_parameters is None:
        hyper_parameters = {'kernel': ['rbf'], 'gamma': [1]}

    if multi_class:
        for el in hyper_parameters:
            el['decision_function_shape'] = ['ovo']

    normalizer = normalizer_function()
    x_train = normalizer.fit_transform(data)

    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(classifier(), hyper_parameters, cv=5, scoring=score, n_jobs=6, verbose=5)
    clf.fit(x_train, labels)

    return clf


def grid_search(data, labels, test_size=0.25, multi_class=False,
                normalizer_function=MinMaxScaler, classifier=SVC, hyper_parameters=None):

    if hyper_parameters is None:
        hyper_parameters = {'kernel': ['rbf'], 'gamma': [1]}

    if multi_class:
        for el in hyper_parameters:
            el['decision_function_shape'] = ['ovo']

    # Split the data set in two equal parts
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=None)

    normalizer = normalizer_function()
    normalizer.fit(x_train)
    x_train = normalizer.transform(x_train)
    x_test = normalizer.transform(x_test)

    for score in ['accuracy']:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(classifier(), hyper_parameters, cv=5, scoring=score, n_jobs=6, verbose=5)
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


if __name__ == '__main__':

    test_rate = float(sys.argv[1])
    saccade = sys.argv[2] == 's'
    sbj_normalize = sys.argv[3] == 'n'

    out_name = 'results'

    if saccade:
        data_path = '/Users/alerossi/Downloads/jupyter_sketches/output_fixESac.csv'
        out_name += '_saccade'
    else:
        data_path = '/Users/alerossi/Downloads/jupyter_sketches/output.csv'
        out_name += '_norm'

    # data_path = 'output.csv'
    class_column = 'image'
    """
            SETTING TEST PARAMETERS
    
    """
    trials = 10

    scores = [
        'accuracy',
        'f1'
    ]

    out_cols = ['Test', 'Normalization', 'Classifier', 'Min_length', 'Accuracy', 'F1', 'parameters']

    if test_rate == 0:
        trials = 1
        out_name += '_notest'
        out_cols = ['Test', 'Normalization', 'Classifier', 'Min_length']
        if 'accuracy' in scores:
            out_cols += ['Accuracy_Avg', 'Accuracy_Std', 'Accuracy_parameters']
        if 'f1' in scores:
            out_cols += ['F1_Avg', 'F1_Std', 'F1_parameters']

    out_name += '.csv'
    normalizers = {
        'minmax': MinMaxScaler,
        'zmean': StandardScaler
    }

    classifier_to_test = {
        'svm': SVC,
        'log-reg': LogisticRegression,
        'RF': RandomForestClassifier,
        'NN': MLPClassifier
    }

    # Set the parameters by cross-validation
    gamma = ['auto', 'scale']#[100,  10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5] # 8
    c_weight = [0.001, 0.01, 0.1, 1, 10, 20] # 11
    deg = [3, 6, 9]
    class_weight = [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}]#, {0: 1, 1: 3}, {0: 3, 1: 1}, {0: 1, 1: 4}, {0: 4, 1: 1}] #7
    lr = [0.01, 0.001]
    hu = [(16,), (32,), (64,), (128,), (32, 32,), (64, 64,)]
    act = ['tanh', 'relu']
    alpha = [1e-5, 1e-4, 1e-3, 1e-2]
    max_iter = [10000]

    hyper_parameters_to_test = {
        'svm': [
            {'kernel': ['rbf'], 'gamma': gamma, 'C': c_weight, 'class_weight': class_weight},
            {'kernel': ['linear'], 'C': c_weight, 'class_weight': class_weight},
            {'kernel': ['poly'], 'degree': deg, 'gamma': gamma, 'C': c_weight, 'class_weight': class_weight}
        ],
        'log-reg': [{'penalty': ['l1', 'l2'], 'C': c_weight, 'class_weight': class_weight}],
        'RF': [{'n_estimators': [10], 'criterion': ['gini', 'entropy'], 'class_weight': class_weight}],
        'NN': [
            {
                'hidden_layer_sizes': hu, 'activation': act, 'alpha': alpha, 'max_iter': max_iter,
                'solver': ['sgd'], 'learning_rate': ['invscaling', 'constant'],
                'momentum': [.1, .9], 'nesterovs_momentum': [True],
                'learning_rate_init': lr
            },
            {
                'hidden_layer_sizes': hu, 'activation': act, 'alpha': alpha, 'max_iter': max_iter,
                'solver': ['adam'], 'learning_rate_init': lr
            }
        ]
    }

    svm_trials = (len(gamma) * len(c_weight) * len(class_weight)) * (len(deg) + 1) + len(c_weight) * len(class_weight)
    log_trials = len(c_weight) * len(class_weight) * 2
    rf_trials = len(class_weight) * 2
    class_trials = svm_trials + log_trials + rf_trials

    """
            CREATING DATA SET
    
    """
    data_cols = [
        'n_fix',
        # 'fix_max', 'fix_mean',
        'norm_fix_max', 'norm_fix_mean'
    ]
    if saccade:
        data_cols += [
            'x_regressions', 'y_regressions',
            'up_freq', 'down_freq', 'left_freq', 'right_freq',
            'min_duration', 'avg_duration', 'max_duration',
            'min_vel', 'avg_vel', 'max_vel',
            'min_ampl', 'avg_ampl', 'max_ampl',
            'min_angle', 'avg_angle', 'max_angle',
            'min_distance', 'avg_distance', 'max_distance',
            'min_slope', 'avg_slope', 'max_slope'
        ]
    grouped_var = {
        'NR': ['NWoI', 'RWoI'],
        'N': ['NWoI', 'NWI'],
        'R': ['RWoI', 'RWI']
    }

    df = pd.read_csv(data_path)

    if sbj_normalize:
        for c_var in data_cols:
            for i in range(len(df)):
                df[c_var].iloc[i] -= df[df['RECORDING_SESSION_LABEL'] == df['RECORDING_SESSION_LABEL'].iloc[i]][c_var].mean()

    dataset = {}
    for k, v in grouped_var.items():
        cdf = df[df[class_column].isin(v)]
        cdf[class_column] = cdf[class_column].astype('category')

        dataset[k] = cdf.copy()

    total_cases = class_trials * len(normalizers) * len(grouped_var) * len(cdf['minimum_length'].unique()) * trials
    print(total_cases)

    results = []
    for ck, cv in classifier_to_test.items():
        print("\n\n  ---   Classifier: {}".format(ck))
        for nk, nv in normalizers.items():
            print("\n\n  ---   Normalizer: {}".format(nk))
            for k, v in dataset.items():
                print("\n\n     -----     Data type: {}\n\n".format(k))

                for lv in v['minimum_length'].unique():
                    print("\n\n     -----     Min Length: {}\n\n".format(lv))

                    curr_len_df = v[v['minimum_length'] == lv]
                    for t in range(trials):
                        print("\n        -------        Trial: {}\n".format(t))

                        if test_rate > 0:
                            tclf, [_, x_test, _, y_test] = grid_search(
                                data=curr_len_df[data_cols],
                                labels=curr_len_df[class_column].cat.codes.values,
                                test_size=test_rate,
                                multi_class=False,
                                normalizer_function=nv,
                                classifier=cv,
                                hyper_parameters=hyper_parameters_to_test[ck]
                            )
                            # acc = compute_confusion(tclf, out_data)
                            y_pred = tclf.predict(x_test)
                            results.append([
                                k, nk, ck, lv,
                                accuracy_score(y_test, y_pred), f1_score(y_test, y_pred),
                                str(tclf.best_params_)])

                        else:
                            new_results_row = [k, nk, ck, lv]
                            for score in scores:
                                tclf = grid_search_notest(
                                    data=curr_len_df[data_cols],
                                    labels=curr_len_df[class_column].cat.codes.values,
                                    multi_class=False,
                                    normalizer_function=nv,
                                    score=score,
                                    classifier=cv,
                                    hyper_parameters=hyper_parameters_to_test[ck]
                                )
                                best_score_avg = tclf.cv_results_['mean_test_score'].max()
                                b_index = tclf.cv_results_['mean_test_score'].argmax()
                                best_score_std = tclf.cv_results_['std_test_score'][b_index]
                                new_results_row.append(best_score_avg)
                                new_results_row.append(best_score_std)
                                new_results_row.append(str(tclf.best_params_))

                            results.append(new_results_row)

                    results_df = pd.DataFrame(data=results, columns=out_cols)
                    results_df.to_csv(out_name, index=False)

    if test_rate > 0:
        g_var = ['Test', 'Normalization', 'Classifier', 'Min_length']
        s_var = {'Accuracy': 'avg_Acc', 'F1': 'avg_F1'}
        gr = results_df.groupby(g_var)[list(s_var.keys())].mean().rename(columns=s_var)
        gr[['std_Acc', 'std_F1']] = results_df.groupby(g_var)[list(s_var.keys())].std()

        gr.reset_index().to_csv('results_stats.csv', index=False)
