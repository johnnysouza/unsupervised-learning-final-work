import cv2
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

ORL_PATH = f'data{os.path.sep}ORL'

def load_dataset_file(file_path: str):
    file_data = cv2.cvtColor(cv2.imread(os.path.join(ORL_PATH, file_path)), cv2.COLOR_BGR2GRAY)
    label = int(file_path.split('.')[0].split('_')[-1])
    return file_data, label
        

def pca():
    X, y = zip(*[load_dataset_file(file) for file in os.listdir(ORL_PATH)])
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(X_train, y_train)

    for i, x_test in enumerate(X_test):
        [predicted_label, predicted_conf] = recognizer.predict(x_test)
        print(f'Predicted label {predicted_label}')
        print(f'True label {y_test[i]}')


if __name__ == '__main__':
    pca()