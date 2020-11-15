import os

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

ORL_PATH = f'..{os.path.sep}data{os.path.sep}ORL'

def load_dataset_file(file_path: str):
    with open(os.path.join(ORL_PATH, file_path), 'rb') as file:
        file_data = file.read()
        yield file_data, file_path.split('.')[0].split('_')[-1]

def pca():
    X, y = [load_dataset_file(file) for file in os.listdir(ORL_PATH)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
    print(y_test)


def __main__():
    pca()