import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class UtilityFunctions:
    """
    This class contains utility functions
    """

    @staticmethod
    def generate_checkerboard(rows: int, cols: int) -> pd.DataFrame:
        """
        This function generates a Pandas dataframe object describing a checkerboard.
        :param rows: Number of rows
        :param cols: Number of columns
        :return: Checkerboard as pandas dataframe
        """

        squares = []

        for current_row in range(0, rows):
            for current_col in range(0, cols):
                current_square = {
                    'x': current_col,
                    'y': current_row,
                    'true_color': (current_col + current_row) % 2
                }

                squares.append(current_square)

        return pd.DataFrame(squares)

    @staticmethod
    def visualize_simple_checkerboard(checkerboard: pd.DataFrame, output: str = None):
        """
        This function visualizes a given checkerboard.
        :param checkerboard: Dataframe representing the checkerboard
        :param output: Relative path of the output image (optional)
        """

        ax = plt.gca()

        ax.set_xticks(np.arange(0, max(checkerboard['x'] + 1), 1))
        ax.set_yticks(np.arange(0, max(checkerboard['y'] + 1), 1))

        plt.imshow(checkerboard.pivot('y', 'x', 'true_color'), origin='lower', cmap='copper', aspect='equal')

        if output:
            plt.savefig(output)
            plt.clf()
        else:
            plt.show()
            plt.clf()

    @staticmethod
    def visualize_extrapolated_checkerboard(training_board: pd.DataFrame, extrapolated_board: pd.DataFrame, title: str, f1_score: float,
                                            output: str = None):
        """
        This function visualizes an extrapolated checkerboard.
        :param training_board: A pandas dataframe representing the training board
        :param extrapolated_board: A pandas dataframe representing the predicted section of the extrapolated board.
        :param title: Title of the plot. Model name will be passed here.
        :param f1_score: F1-score of the utilized model.
        :param output: Relative path of the output image.
        :return:
        """
        ax = plt.gca()

        ax.set_xticks(np.arange(0, max(extrapolated_board['x'] + 1), 1))
        ax.set_yticks(np.arange(0, max(extrapolated_board['y'] + 1), 1))

        plt.imshow(training_board.pivot('y', 'x', 'true_color'), origin='lower', cmap='copper', aspect='equal')
        plt.imshow(extrapolated_board.pivot('y', 'x', 'predicted_color'), origin='lower', cmap='summer', aspect='equal')

        ax.set_title(f'{title}\nF1-score:{f1_score}')

        if output:
            plt.savefig(output)
            plt.clf()
        else:
            plt.show()
            plt.clf()

    @staticmethod
    def subtract_boards(minuend: pd.DataFrame, subtrahend: pd.DataFrame) -> pd.DataFrame:
        """
        This utility function calculates the difference of two checkerboards for test purposes.
        :param minuend: Minuend (bigger) checkerboard
        :param subtrahend: Subtrahend (smaller) checkerboard
        :return: Difference checkerboard
        """

        return pd.concat([minuend, subtrahend]).drop_duplicates(keep=False)
