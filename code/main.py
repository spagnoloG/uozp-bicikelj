import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
from enum import Enum
from tqdm import tqdm
import torch.utils.data as data
import numpy as np
from datetime import datetime, timedelta


class Scoring(Enum):
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"
    NEG_MEAN_ABSOLUTE_ERROR = "neg_mean_absolute_error"
    NEG_MEAN_SQUARED_ERROR = "neg_mean_squared_error"


# class LSTM(nn.Module):
#    def __init__(self, input_size=5, hidden_size=40, num_layers=1,
#                 batch_first=True, optimizer='adam', lr=0.01, epochs=2000,
#                 loss_fn=nn.MSELoss()):
#        super(LSTM, self).__init__()
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.num_layers = num_layers
#        self.batch_first = batch_first
#        self.lr = lr
#        self.epochs = epochs
#        self.loss_fn = loss_fn
#
#        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first)
#        self.fc1 = nn.Linear(40, 1)
#
#        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#    def forward(self, x):
#        x, _ = self.lstm(x)
#        x = self.fc1(x)
#        return x
#
#
#    def train_eval(self, X_train, y_train, X_test, y_test):
#        self.to(self.device)
#        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
#        X_test, y_test = X_test.to(self.device), y_test.to(self.device)
#
#        self.train()
#        optimizer = optim.Adam(self.parameters(), lr=self.lr)
#        loader = DataLoader(data.TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
#
#        for epoch in range(self.epochs):
#            self.train()
#            for x_batch, y_batch in loader:
#                y_pred = self(x_batch)
#                loss = self.loss_fn(y_pred, y_batch)
#                optimizer.zero_grad()
#                loss.backward()
#                optimizer.step()
#            if epoch % 5 == 0:
#                self.eval()
#                y_pred = self(X_train)
#                train_loss = self.loss_fn(y_pred, y_train)
#                y_pred = self(X_test)
#                test_loss = self.loss_fn(y_pred, y_test)
#                print(f'Epoch {epoch} train loss: {train_loss} test loss: {test_loss}')
#
#
#        self.weights = self.state_dict()
#
#    def predict(self, X):
#        self.to(self.device)
#        X = X.to(self.device)
#
#        self.eval()
#        with torch.no_grad():
#            y_pred = self(X)
#        return y_pred.cpu().numpy()
#


class EvaluateModel(BaseEstimator, RegressorMixin):
    def __init__(self, model, hyper_params):
        self.model = model
        self.hyper_params = hyper_params
        self.score: dict = None
        self.best_params_grid_search: dict = None

    def fit(self, X, y):
        grid_search = RandomizedSearchCV(
            self.model,
            self.hyper_params,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=4,
        )
        grid_search.fit(X, y)
        self.score = grid_search.best_score_
        self.best_params_grid_search = grid_search.best_params_
        self.model = grid_search.best_estimator_

        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


def plot_station_over_time(train_set_path: str, station_name: str):
    train_set = pd.read_csv(train_set_path)
    try:
        train_set = train_set[["timestamp", station_name]]
    except KeyError:
        raise KeyError("Station name not found in the dataset.")

    # Convert the timestamp to a datetime object
    train_set["timestamp"] = pd.to_datetime(train_set["timestamp"])

    # Now plot the values of the station name over time
    train_set.plot(x="timestamp", y=station_name, figsize=(20, 10))
    plt.show()


def create_dataset(X, y, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X_r, y_r = [], []
    for i in range(len(X) - lookback):
        feature = X[i : i + lookback]
        target = y[i + 1 : i + lookback + 1]
        X_r.append(feature)
        y_r.append(target)
    return torch.tensor(np.array(X_r), dtype=torch.float32), torch.tensor(
        np.array(y_r), dtype=torch.float32
    )


# def load_train_dataset(train_set_path: str, lookback: int = 3):
#    """
#    Loads the data from the given paths and applies windowing.
#    :param train_set_path: Path to the train set.
#    :param lookback: Size of window for prediction.
#    :return: List of datasets, where each dataset is a tuple (X_train, X_test, y_train, y_test).
#    """
#    train_set = pd.read_csv(train_set_path)
#    timestamps = train_set["timestamp"]
#    date_dataframe = pd.DataFrame()
#    date_dataframe["day"] = pd.to_datetime(timestamps).dt.day
#    date_dataframe["month"] = pd.to_datetime(timestamps).dt.month
#    date_dataframe["hour"] = pd.to_datetime(timestamps).dt.hour
#    date_dataframe["minute"] = pd.to_datetime(timestamps).dt.minute
#    date_dataframe["day_of_week"] = pd.to_datetime(timestamps).dt.dayofweek
#
#    datasets = []
#    for col in train_set.columns:
#        if col != "timestamp":
#
#            dataset = train_set[col].to_frame().join(date_dataframe)
#            # Get the first column
#            target_var_name = dataset.columns[0]
#            # Split the data into train and test time series
#            X = dataset.drop([target_var_name], axis=1)
#            y = dataset[target_var_name]
#
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
#            X_train, y_train = create_dataset(X_train.values, y_train.values, lookback)
#            X_test, y_test = create_dataset(X_test.values, y_test.values, lookback)
#
#            datasets.append((X_train, X_test, y_train, y_test))
#
#    return datasets


# Next improvement would be to find the closest two stations to the one we are predicting and use their data
def get_historic_data(station_name, X, lookback=3):
    all_data = pd.read_csv("../data/bicikelj_train.csv")
    all_data["timestamp"] = pd.to_datetime(
        all_data["timestamp"], format="%Y-%m-%d %H:%M:%S"
    )

    X["timestamp"] = pd.to_datetime(X["timestamp"], format="%Y-%m-%d %H:%M:%S")

    for timestamp in range(30, 121, 30):
        # Create a copy of all_data and modify the timestamps
        shifted_all_data = all_data.copy()
        shifted_all_data["timestamp"] = shifted_all_data["timestamp"] + timedelta(
            minutes=timestamp
        )

        # Merge X with the shifted_all_data using merge_asof, which finds the closest match
        merged_data = pd.merge_asof(
            X, shifted_all_data, on="timestamp", direction="nearest"
        )

        # Set the attribute value
        X[f"attr_y{timestamp}"] = merged_data[station_name]

    return X


def load_and_preprocess_data(
    train_set_path: str, test_set_path: str, metadata_path: str
):
    metadata = pd.read_csv(metadata_path)
    train_set = pd.read_csv(train_set_path)
    test_set = pd.read_csv(test_set_path)

    def standardize(df_train, df_test):
        scaler = StandardScaler()
        scaler.fit(df_train)
        df_train = scaler.transform(df_train)
        df_test = scaler.transform(df_test)

        return df_train, df_test

    def split_dataset(df):
        X = df[["timestamp"]].copy()

        y = df.drop(["timestamp"], axis=1).copy()

        return X, y

    stations = []
    for st_train, st_test in zip(train_set.columns, test_set.columns):
        if st_train != st_test:
            raise ValueError("Columns in train and test set do not match.")
        if st_train == "timestamp":
            continue

        station_train_set = train_set[["timestamp", st_train]].copy()
        station_test_set = test_set[["timestamp", st_test]].copy()

        # print(station_train_set.head())

        X_train, y_train = split_dataset(station_train_set)
        X_test, y_test = split_dataset(station_test_set)

        # For each timestamp in X_train, locate the corresponding metadata
        # and add it to the dataframe

        X_train = pd.merge(X_train, metadata, on="timestamp", how="left")
        X_test = pd.merge(X_test, metadata, on="timestamp", how="left")

        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # X_train = X_train.merge(train_set)
        # X_test = X_test.merge(test_set)

        X_train = get_historic_data(st_train, X_train, lookback=3)
        X_test = get_historic_data(st_test, X_test, lookback=3)
        timestamp_test_df = X_test["timestamp"].copy()
        X_test.drop(["timestamp"], axis=1, inplace=True)
        X_train.drop(["timestamp"], axis=1, inplace=True)

        X_train, X_test = standardize(X_train, X_test)

        # X_train, X_test = standardize(X_train, X_test)
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # Count the nan values
        # print(X_train.isna().sum())
        # print(X_test.isna().sum())

        stations.append((st_train, timestamp_test_df, X_train, X_test, y_train, y_test))

    return stations


def main():
    TRAIN_PHASE = False

    if TRAIN_PHASE:
        stations = load_and_preprocess_data(
            "../data/bicikelj_train_indices.csv",
            "../data/bicikelj_test_indices.csv",
            "../data/bicikelj_metadata.csv",
        )

        rf_params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 9, 11, 13, 15],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "bootstrap": [True, False],
            "ccp_alpha": [0.0, 0.1, 0.2],
        }

        print(rf_params)

        avg_mse = 0
        for station in stations:
            st_train, timestamp_test_df, X_train, X_test, y_train, y_test = station

            reg = LinearRegression().fit(X_train, y_train)
            # rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=0)

            # em = EvaluateModel(model=RandomForestRegressor(), hyper_params=rf_params)

            y_train = y_train.values.ravel()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            y_pred = np.rint(y_pred)
            print("coefs:", reg.coef_)

            print("MAE:", mean_absolute_error(y_test, y_pred))
            avg_mse = avg_mse + mean_absolute_error(y_test, y_pred)

        print("Average MAE:", avg_mse / len(stations))

    else:
        stations = load_and_preprocess_data(
            "../data/bicikelj_train.csv",
            "../data/bicikelj_test.csv",
            "../data/bicikelj_metadata.csv",
        )

        _, result, _, _, _, _ = stations[0]

        for station in stations:
            st_train, _, X_train, X_test, y_train, y_test = station
            print("training station:", st_train)

            y_train = y_train.values.ravel()
            reg = LinearRegression().fit(X_train, y_train)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            y_pred = np.rint(y_pred)

            # Set all negative values to 0
            y_pred[y_pred < 0] = 0

            # Convert y_pred to dataframe
            y_pred_df = pd.DataFrame(y_pred, columns=[st_train])
            result = pd.concat([result, y_pred_df], axis=1)

        result.to_csv("../data/bicikelj_result.csv", index=False)

    # X_train, y_train = create_dataset(X_train, y_train, 3)
    # X_test, y_test = create_dataset(X_test, y_test, 1)

    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # lstm.train_eval(X_train, y_train, X_test, y_test)

    # datasets = load_train_dataset("../data/bicikelj_train.csv")

    # X_train, X_test, y_train, y_test = datasets[0]
    # print(X_train, y_train)

    # lm = LSTM()
    # lm.train_eval(X_train, y_train, X_test, y_test)

    # plot_station_over_time("../data/bicikelj_train.csv", "GERBIČEVA - ŠPORTNI PARK SVOBODA")

    # datasets = load_and_preprocess_data("../data/bicikelj_train_indices.csv", "../data/bicikelj_test_indices.csv")

    # for dataset in datasets:
    #    X_train, X_test, y_train, y_test = dataset
    #    model = EvaluateModel(LinearRegression(), {"fit_intercept": [True, False]})
    #    model.fit(X_train, y_train)
    #    print(model.predict(X_test))


if __name__ == "__main__":
    main()
