import requests
import pandas as pd

from sklearn.datasets import fetch_mldata
import sys

def progress_bar(progress, text):
    """
    Prints progress bar to console
    Args:
        progress: float in [0,1] representing progress in action
                    where 0 nothing done and 1 completed.
        text: Short string to add after progress bar.
    """
    if isinstance(progress, int):
        progress = float(progress)
    block = int(round(20*progress))
    progress_line = "\rCompleted: [{0}] {1:5.2f}% {2}.".format("#"*block + "-"*(20-block), progress*100, text)
    sys.stdout.write(progress_line)
    sys.stdout.flush()

def load_mnist():
    try:
        mnist = fetch_mldata('MNIST original')

        data_df = pd.DataFrame(mnist['data'])
        label_df = pd.DataFrame(mnist['target'])

        return data_df, label_df
    except requests.exceptions.RequestException:
        print('HTTP exception, check you connection and try again')
