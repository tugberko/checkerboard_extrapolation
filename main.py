import os.path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utility_functions import UtilityFunctions
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def extrapolate(training_checkerboard: pd.DataFrame, target_rows: int, target_cols: int, model):
    """
    This function extrapolates the given training checkerboard to desired dimensions.
    :param training_checkerboard: Pandas dataframe representing the training board.
    :param target_rows: Number of rows in the target board.
    :param target_cols: Number of columns in the target board
    :param model: The model intended to be used for the extrapolation.
    """

    # Fit the training board first.
    model.fit(training_checkerboard[['x', 'y']], training_checkerboard['true_color'])

    target_checkerboard = UtilityFunctions.generate_checkerboard(target_rows, target_cols)

    extrapolated_checkerboard = UtilityFunctions.subtract_boards(target_checkerboard, training_checkerboard)

    # Append a new column of predicted colors
    extrapolated_checkerboard['predicted_color'] = model.predict(extrapolated_checkerboard[['x', 'y']])

    # Calculate the F1 score
    score = f1_score(y_true=extrapolated_checkerboard['true_color'],
                     y_pred=extrapolated_checkerboard['predicted_color'])

    model_name = str(type(model)).split('.')[-1][:-2]

    output_path = os.path.join('outputs', f'{model_name}.png')

    # Visualize the extrapolated checkerboard
    UtilityFunctions.visualize_extrapolated_checkerboard(training_checkerboard, extrapolated_checkerboard,
                                                         title=model_name, f1_score=score, output=output_path)


def main():
    training_checkerboard = UtilityFunctions.generate_checkerboard(5, 5)

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    for current_classifier in classifiers:
        extrapolate(training_checkerboard, 10, 10, current_classifier)


if __name__ == '__main__':
    main()
