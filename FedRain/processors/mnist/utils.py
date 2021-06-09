# Library that loads mnist
from typing import Any, Tuple

import mnist
import numpy as np

# Library for converting mnist images to floats
from skimage import img_as_float
from skimage.filters import gaussian
from sklearn.linear_model import LogisticRegression

# MNIST has a prespecified train/test split
# But is a "relatively" big dataset (at least for now)
# We use train_test_split for subsampling
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.framework.dtypes import DType


def load_dataset(
    seed: int, train_size: int = 3000, test_size: int = 3000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the mnist dataset, the label is not one hot encoded yet
    """

    Xtrain = mnist.train_images().reshape(-1, 784)
    Xtest = mnist.test_images().reshape(-1, 784)
    ytrain = mnist.train_labels().astype(np.int64)
    ytest = mnist.test_labels().astype(np.int64)

    X = np.concatenate([Xtrain, Xtest])
    y = np.concatenate([ytrain, ytest])
    # For now we do not support very large datasets

    # It is important that these subsets are not affected by the seed
    # So that the seed only affects the corruption (not the data split)

    # Let us subsample the training set
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=seed
    )

    # Transform to floats
    Xtrain = img_as_float(Xtrain)
    Xtest = img_as_float(Xtest)

    return Xtrain, ytrain, Xtest, ytest


def single_digit_corruption(
    y: np.ndarray, corrupted_digit: int, corrupted_to_digit: int, corruption_rate: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Corrupt the `corrupted_digit` with `corrupted_to_digit`, 
    returns the corrupted labels and the selector on which sample is corrupted
    """
    # Let us copy the clean labels
    assert len(y.shape) == 1

    ycorr = y.copy()

    # Let us identify corruption candidates
    candidates, = np.where(y == corrupted_digit)
    # Let us find how many corruptions to do via rounding
    corruptions = int(candidates.shape[0] * corruption_rate)
    # Let us sample without replacement $corruptions candidates
    np.random.seed(seed)
    corrupt_idx = np.random.choice(candidates, size=corruptions, replace=False)
    corrsel = np.full((len(y),), False)
    corrsel[corrupt_idx] = True

    # And switch the labels of those to $self.corrupted_to_class
    ycorr[corrupt_idx] = corrupted_to_digit
    return ycorr, corrsel


def blur_annotator_corruption(
    Xpre: np.ndarray, ypre: np.ndarray, X: np.ndarray, y: np.ndarray, sigma: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Corrupt the labels using Short Eyesight Annotator corruption from DUTI
    """
    annotator = LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=seed, multi_class="auto"
    ).fit(Xpre, ypre)

    Xcopy = X.copy()

    for i in range(len(Xcopy)):
        Xcopy[i] = gaussian(Xcopy[i].reshape(28, 28), sigma=sigma).reshape(-1)
    predicted = annotator.predict(Xcopy)
    corrsel = y != predicted
    return predicted, corrsel
