
import os
import argparse
import itertools
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from azureml.core import Dataset, Run
run = Run.get_context()


def log_confusion_matrix_image(cm, labels, normalize=False, log_name="confusion_matrix", title="Confusion matrix", cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    run.log_image(log_name, plot=plt)
    plt.savefig(os.path.join("outputs", "{0}.png".format(log_name)))


def log_confusion_matrix(cm, labels):
    # log confusion matrix as object
    cm_json = {
        "schema_type": "confusion_matrix",
        "schema_version": "v1",
        "data": {
            "class_labels": labels,
            "matrix": cm.tolist()
        }
    }
    run.log_confusion_matrix("confusion_matrix", cm_json)

    # log confusion matrix as image
    log_confusion_matrix_image(cm, labels, normalize=False, log_name="confusion_matrix_unnormalized", title="Confusion matrix")

    # log normalized confusion matrix as image
    log_confusion_matrix_image(cm, labels, normalize=True, log_name="confusion_matrix_normalized", title="Normalized confusion matrix")


def main(args):
    
    print("... start train.main  ...")
    # Create the outputs folder
    os.makedirs(name="outputs", exist_ok=True)
    
    # Log arguments
    run.log(name="Kernel type", value=np.str(args.kernel))
    run.log(name="Penalty", value=np.float(args.penalty))

    # Load iris dataset
    dataset = run.input_datasets["iris"]

    #ws = run.experiment.workspace 
    # get the input dataset by ID 
    #dataset = Dataset.get_by_id(ws, id=args.input_data)

    try:
        # try to load tabular dataset
        df = dataset.to_pandas_dataframe()
    except:
        # try to load mounted file dataset
        print("Dataset path: ", str(dataset))
        df = pd.read_csv(os.path.join(dataset))
    
    # split dataset
    x_col = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    y_col = ["species"]
    x_df = df.loc[:, x_col]
    y_df = df.loc[:, y_col]
    
    #dividing X,y into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=223)
    data = {"train": {"X": x_train, "y": y_train},
            "test": {"X": x_test, "y": y_test}}

    # labels
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    
    # train a SVM classifier
    svm_model = SVC(
        C=args.penalty,
        kernel=args.kernel,
        degree=3,
        gamma="scale",
        probability=True
    )
    svm_model = svm_model.fit(X=data["train"]["X"], y=data["train"]["y"])
    svm_predictions = svm_model.predict(X=data["test"]["X"])

    # accuracy for X_test
    accuracy = svm_model.score(X=data["test"]["X"], y=data["test"]["y"])
    print("Accuracy of SVM classifier on test set: {:.2f}".format(accuracy))
    run.log(name="Accuracy", value=np.float(accuracy))

    # precision for X_test
    precision = precision_score(y_true=data["test"]["y"], y_pred=svm_predictions, labels=labels, average="weighted")
    print("Precision of SVM classifier on test set: {:.2f}".format(precision))
    run.log(name="precision", value=precision)

    # recall for X_test
    recall = recall_score(y_true=data["test"]["y"], y_pred=svm_predictions, labels=labels, average="weighted")
    print("Recall of SVM classifier on test set: {:.2f}".format(recall))
    run.log(name="recall", value=recall)

    # f1-score for X_test
    f1 = f1_score(y_true=data["test"]["y"], y_pred=svm_predictions, labels=labels, average="weighted")
    print("F1-Score of SVM classifier on test set: {:.2f}".format(f1))
    run.log(name="f1-score", value=f1)

    # create a confusion matrix
    cm = confusion_matrix(y_true=data["test"]["y"], y_pred=svm_predictions, labels=labels)
    log_confusion_matrix(cm, labels)

    # files saved in the "outputs" folder are automatically uploaded into run history
    model_file_name = "model.pkl"
    joblib.dump(value=svm_model, filename=os.path.join("outputs", model_file_name))


def parse_args():
    parser = argparse.ArgumentParser(description="Model pareameter args")
    parser.add_argument("--kernel", type=str, default="rbf", required=False, help="Kernel type to be used in the algorithm", dest="kernel")
    parser.add_argument("--penalty", type=float, default=1.0, required=False, help="Penalty parameter of the error term", dest="penalty")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
