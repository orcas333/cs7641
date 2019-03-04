import sklearn
import sklearn.svm
import sklearn.neighbors
import sklearn.tree
import sklearn.ensemble
import sklearn.neural_network
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
import logging
from sklearn import datasets
from tensorflow import keras
import copy
import pandas as pd
import matplotlib.pyplot as plt

def import_digit_data():
    logging.info("Importing Digits Data...")
    digits = datasets.load_digits()
    return digits

def tranche_digit_data(n, input_data):

    digits = copy.deepcopy(input_data)

    if n is not False:
        digits.images = digits.images[:n, :, :]
        digits.target = digits.target[:n]

    logging.info("Length of dataset is {}".format(len(digits.target)))
    n_samples = len(digits.images)
    digit_data = digits.images.reshape((n_samples, -1))

    train_images = digit_data[:n_samples // 2]
    train_labels = digits.target[:n_samples // 2]
    test_images = digit_data[n_samples // 2:]
    test_labels = digits.target[n_samples // 2:]

    return train_images, train_labels, test_images, test_labels

def tranche_digit_data_short(n, input_data):

    digits = copy.deepcopy(input_data)

    if n is not False:
        digits.images = digits.images[:n, :, :]
        digits.target = digits.target[:n]

    logging.info("Length of dataset is {}".format(len(digits.target)))
    n_samples = len(digits.images)
    digit_data = digits.images.reshape((n_samples, -1))

    train_images = digit_data
    train_labels = digits.target

    return train_images, train_labels

def import_fashion_data():
    logging.info("Importing Keras Data...")
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    return train_images, train_labels, test_images, test_labels

def tranche_fashion_data(n, train_images, train_labels, test_images, test_labels):

    if n is not False:
        train_images = train_images[:n, :, :]
        train_labels = train_labels[:n]
        test_images = test_images[:n, :, :]
        test_labels = test_labels[:n]

    logging.info("Length of dataset is {}".format(len(train_labels)))
    train_data = train_images.reshape((len(train_images), -1))
    test_data = test_images.reshape((len(test_images), -1))

    train_images = train_data
    test_images = test_data

    return train_images, train_labels, test_images, test_labels

def tranche_fashion_data_short(n, train_images, train_labels):

    if n is not False:
        train_images = train_images[:n, :, :]
        train_labels = train_labels[:n]

    logging.info("Length of dataset is {}".format(len(train_labels)))
    train_data = train_images.reshape((len(train_images), -1))
    train_images = train_data

    return train_images, train_labels

def run_multi_samples(data_type):

    summary = []

    logging.info("Running simulation for {} dataset".format(data_type))

    if data_type in ['fashion']:
        input_train_images, input_train_labels, input_test_images, input_test_labels = import_fashion_data()
        sample_size_space = [100, 200, 500, 700, 1000, 1200, 1500, 2000, 5000, 7000, 10000]

    if data_type in ['digits']:
        run_data = import_digit_data()
        sample_size_space = [100, 200, 500, 700, 1000, 1200, 1500, 2000]

    for n in sample_size_space:
        input_results_dict = {'dataset_name': data_type,
                      'num_samples': n,
                      'n_fold': 0}

        if data_type in ['fashion']:
            train_images = copy.deepcopy(input_train_images)
            train_labels = copy.deepcopy(input_train_labels)
            test_images = copy.deepcopy(input_test_images)
            test_labels = copy.deepcopy(input_test_labels)

            train_images, train_labels, test_images, test_labels = tranche_fashion_data(n, train_images, train_labels, test_images, test_labels)

        if data_type in ['digits']:
            train_images, train_labels, test_images, test_labels = tranche_digit_data(n, run_data)

            train_images_df = pd.DataFrame(train_images)
            train_labels_df = pd.DataFrame(train_labels)
            test_images_df = pd.DataFrame(test_images)
            test_labels_df = pd.DataFrame(test_labels)
            train_images_df.to_csv('digits_train_images.csv')
            train_labels_df.to_csv('digits_train_labels.csv')
            test_images_df.to_csv('digits_test_images.csv')
            test_labels_df.to_csv('digits_test_labels.csv')
            print (train_labels)
            print (test_images)
            print (test_labels)

        classifiers = create_models()
        for key, value in classifiers.items():
            results_dict = classify(key, value, train_images, train_labels, test_images, test_labels, input_results_dict)
            summary.append(results_dict)

    summary_df = pd.DataFrame(summary)
    logging.info("Outputing .csv result data...")
    summary_df.to_csv('{}_learning_curve_data.csv'.format(data_type))

def classify(model_id, model, train_images, train_labels, test_images, test_labels, input_results_dict, n_folds=False):
    model_type, accuracy_training, accuracy_validation = train_and_evaluate_model(model, train_images, train_labels, test_images, test_labels)
    results_dict = copy.deepcopy(input_results_dict)
    results_dict['model_id'] = model_id
    results_dict['model_type'] = model_type
    results_dict['accuracy_training'] = accuracy_training
    results_dict['accuracy_validation'] = accuracy_validation

    if n_folds is not False:
        results_dict['n_fold'] = n_folds

    return results_dict

def run_multi_cross_validation_samples(data_type, n_folds=2):

    # data_type = 'digits'
    summary = []
    logging.info("Running simulation for {} dataset".format(data_type))

    if data_type in ['fashion']:
        input_train_images, input_train_labels, input_test_images, input_test_labels = import_fashion_data()
        sample_size_space = [200, 500, 700, 1000, 1200, 1500, 2000, 5000, 7000, 10000]

    if data_type in ['digits']:
        run_data = import_digit_data()
        sample_size_space = [200, 500, 700, 1000, 1200, 1500, 2000]

    for n in sample_size_space:
        input_results_dict = {'dataset_name': data_type,
                              'num_samples': n}

        if data_type in ['fashion']:
            train_images = copy.deepcopy(input_train_images)
            train_labels = copy.deepcopy(input_train_labels)

            data, labels = tranche_fashion_data_short(n, train_images, train_labels)

        if data_type in ['digits']:
            data, labels = tranche_digit_data_short(n, run_data)

        classifiers = create_models()
        for key, value in classifiers.items():

            skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=False)
            for i, (train, test) in enumerate(skf.split(data, labels)):
                logging.info("Running Fold {} / {}".format(i + 1, n_folds))
                results_dict = classify(key, value, data[train], labels[train], data[test], labels[test],
                                        input_results_dict, i)

                summary.append(results_dict)


    summary_df = pd.DataFrame(summary)
    logging.info("Outputing .csv result data...")
    summary_df.to_csv('{}_learning_curve_data_wXvalidation.csv'.format(data_type))

def create_models():

    # create your classifiers using this function
    classifiers = {
        "SVM_constant": sklearn.svm.SVC(C=1000),
        "SVM_linear": sklearn.svm.SVC(kernel="linear", C=0.1),
        "SVM_rbf": sklearn.svm.SVC(kernel="rbf", C=0.1),
        "KNN_one": sklearn.neighbors.KNeighborsClassifier(1),
        "KNN_three": sklearn.neighbors.KNeighborsClassifier(3),
        "KNN_five": sklearn.neighbors.KNeighborsClassifier(5),
        "KNN_ten": sklearn.neighbors.KNeighborsClassifier(10),
        "KNN_twenty": sklearn.neighbors.KNeighborsClassifier(20),
        "KNN_fifty": sklearn.neighbors.KNeighborsClassifier(50),
        "DT_best": sklearn.tree.DecisionTreeClassifier(),
        "DT_best_pruned5": sklearn.tree.DecisionTreeClassifier(criterion="gini", splitter='best', max_leaf_nodes=5, min_samples_leaf=5, max_depth=5),
        "DT_best_pruned10": sklearn.tree.DecisionTreeClassifier(criterion="gini", splitter='best', max_leaf_nodes=10, min_samples_leaf=10, max_depth=10),
        "GRB_low": sklearn.ensemble.GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0),
        "GRB_low_nestimators50": sklearn.ensemble.GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_features=2, max_depth=2, random_state=0),
        "GRB_low_nestimators200": sklearn.ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_features=2, max_depth=2, random_state=0),
        "GRB_standard": sklearn.ensemble.GradientBoostingClassifier(n_estimators=20, learning_rate=0.25, max_features=2, max_depth=2, random_state=0),
        "GRB_high": sklearn.ensemble.GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2,max_depth=2, random_state=0),
        "RDF_low_maxdepth10_maxfeatures1": sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=100, max_features=1),
        "RDF_standard_maxdepth10_maxfeatures1": sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=500, max_features=1),
        "RDF_high_maxdepth10_maxfeatures1": sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=1000, max_features=1),
        "RDF_high_maxdepth10_maxfeatures5": sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=1000, max_features=5),
        "RDF_high_maxdepth10_maxfeatures10": sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=1000, max_features=10),
        "RDF_high_maxdepth5_maxfeatures1": sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=1),
        "RDF_high_maxdepth20_maxfeatures1": sklearn.ensemble.RandomForestClassifier(max_depth=20, n_estimators=1000, max_features=1),
        "NN_small": sklearn.neural_network.MLPClassifier(alpha=1, hidden_layer_sizes=(50, 10)),
        "NN_medium": sklearn.neural_network.MLPClassifier(alpha=1, hidden_layer_sizes=(200, 50)),
        "NN_standard": sklearn.neural_network.MLPClassifier(alpha=1, hidden_layer_sizes=(500, 100)),
        "NN_large": sklearn.neural_network.MLPClassifier(alpha=1, hidden_layer_sizes=(1000, 500)),
        "NN_medium_superlowalpha": sklearn.neural_network.MLPClassifier(alpha=0.01, hidden_layer_sizes=(200, 50)),
        "NN_medium_lowalpha": sklearn.neural_network.MLPClassifier(alpha=0.1, hidden_layer_sizes=(200, 50)),
        "NN_medium_highalpha": sklearn.neural_network.MLPClassifier(alpha=10, hidden_layer_sizes=(200, 50)),
    }

    return classifiers

def train_and_evaluate_model(model, train_data, train_labels, test_data, test_labels):
    model.fit(train_data, train_labels)

    # fit and evaluate here.
    model_type = type(model).__name__
    accuracy_training = model.score(train_data, train_labels)
    accuracy_validation = model.score(test_data, test_labels)

    logging.info("{} - Accuracy score (training): {}".format(model_type, accuracy_training))
    logging.info("{} - Accuracy score (validation): {}".format(model_type, accuracy_validation))

    return model_type, accuracy_training, accuracy_validation

def plot_learning_curves(df, data_type, xvalidation=False):
    # plot data
    fig, ax = plt.subplots(figsize=(15, 7))

    plot_df = copy.deepcopy(
            df[['dataset_name', 'model_id', 'model_type', 'num_samples', 'accuracy_training', 'accuracy_validation']])

    plot_df.groupby(['num_samples', 'model_id']).mean().unstack().plot(ax=ax)
    fig.savefig('Learning_curve_{}_all.png'.format(data_type))

    model_unique_values = df['model_type'].unique()

    for model_type in model_unique_values:
        fig, ax = plt.subplots(figsize=(15, 7))
        plot_df = copy.deepcopy(
            df[['dataset_name', 'model_id', 'model_type', 'num_samples', 'accuracy_training', 'accuracy_validation']])
        print_df = copy.deepcopy(plot_df.loc[plot_df['model_type']== model_type])
        print_df.groupby(['num_samples', 'model_id']).mean().unstack().plot(ax=ax)
        fig.savefig('Learning_curve_{}_{}.png'.format(data_type, model_type))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    crossvalidation_toggle = False # Set to True if would like to run cross validation dataset with 2 folds

    dataset_list = ['digits']

    for name in dataset_list:
        run_multi_samples(name)

        if crossvalidation_toggle:
            run_multi_cross_validation_samples(name)