import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, add_dummy_feature
from sklearn.model_selection import train_test_split
import warnings
from os import path
import requests
from zipfile import ZipFile
from io import BytesIO


def load_student_data(file_path=None, target_dir=None, split=0.25):
    """
    Load the student performance dataset (http://archive.ics.uci.edu/ml/datasets/Student+Performance) from a csv file.
    Returns preprocessed training and testing data: X_train, X_test, y_train, y_test
    Provide the `file_path` argument if the dataset is already downloaded locally, otherwise
    the `target_dir` argument can be used to specify where to put the downloaded file.
    """
    if file_path is None:
        r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip')
        if not r.ok:
            raise RuntimeError('''The archive link for the student dataset is not responding - you can try downloading the
.zip file manually from http://archive.ics.uci.edu/ml/datasets/Student+Performance''')
        target_path = './toy-lmopt-classifiers-demo-data/student-data' if target_dir is None else target_dir
        ZipFile(BytesIO(r.content)).extractall(target_path)
        file_path = path.join(target_path, 'student-mat.csv')
    
    df = pd.read_csv(file_path)
    if len(df.columns) == 1:
        df = pd.read_csv(file_path, sep=';')

    target_data = pd.DataFrame(df.G3).applymap(lambda grade: [-1, 1][grade >= 12])
    
    # Encoding categorical values with numerical labels
    numerical_features = df.drop("G3", axis=1).apply(LabelEncoder().fit_transform)

    # Normalisation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normalised_features = add_dummy_feature(StandardScaler().fit_transform(numerical_features))
    preprocessed_features = pd.DataFrame(normalised_features, columns=[ "intercept" ]+list(numerical_features.columns))

    return train_test_split(np.array(preprocessed_features), np.ravel(target_data), test_size=split)


def load_bioseq_data(data_dir, split=0.25):
    X = pd.read_csv(path.join(data_dir, 'Xtr0.csv'), index_col='Id')
    y = pd.read_csv(path.join(data_dir, 'Ytr0.csv'), index_col='Id')
    ds = X.join(y, on='Id')

    return train_test_split(ds, test_size=split)
