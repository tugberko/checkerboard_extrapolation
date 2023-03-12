import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Utilities:
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
                    'color': (current_col + current_row) % 2
                }

                squares.append(current_square)

        return pd.DataFrame(squares)

    @staticmethod
    def visualize_checkerboard(checkerboard: pd.DataFrame, output: str = None):
        """
        This function visualizes a given checkerboard.
        :param checkerboard: Dataframe representing the checkerboard
        :param output: Relative path of the output image (optional)
        """

        ax = plt.gca()

        ax.set_xticks(np.arange(0, max(checkerboard['x'] + 1), 1))
        ax.set_yticks(np.arange(0, max(checkerboard['y'] + 1), 1))

        plt.imshow(checkerboard.pivot('y', 'x', 'color'), origin='lower', cmap='copper', aspect='equal')

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


buyuk = Utilities.generate_checkerboard(12, 12)
kucuk = Utilities.generate_checkerboard(5, 5)
fark = Utilities.subtract_boards(buyuk, kucuk)

Utilities.visualize_checkerboard(fark)
print(fark)
