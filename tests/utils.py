"""Common testing utilities."""

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def regression_dataset(
    n_samples: int = 2000,
    split_size: float = 0.4,
    n_features: int = 1,
    noise: int = 20,
    seed: int = 1,
    shuffle=False,
):
    """Return a regression dataset.

    Args:
        n_samples: Number of samples.
        split_size: Test split size.
        n_features: Number of features.
        noise: Noise level.
        seed: Random seed.
        shuffle: Shuffle the data.

    Returns:
        X_train: Training data.
        X_test: Test data.
        y_train: Training target.
        y_test: Test target.
    """
    X, y = make_regression(  # type: ignore broken types
        n_samples=n_samples, n_features=n_features, noise=noise, random_state=seed
    )

    data_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    X = data_scaler.fit_transform(X)
    y = target_scaler.fit_transform(y.reshape(-1, 1))
    y = y.reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test


def classification_dataset(
    n_samples: int = 2000,
    split_size: float = 0.4,
    n_features: int = 20,
    seed: int = 1,
    shuffle=False,
):
    """Return a classification dataset.

    Args:
        n_samples: Number of samples.
        split_size: Test split size.
        n_features: Number of features.
        seed: Random seed.
        shuffle: Shuffle the data.

    Returns:
        X_train: Training data.
        X_test: Test data.
        y_train: Training target.
        y_test: Test target.
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=seed
    )

    data_scaler = StandardScaler()

    X = data_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test
