import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import argparse
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from tqdm import tqdm
# import logger and setup file to log
import logging
import sys
# Import xgboost
import xgboost as xgb

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Setup file handler
fh = logging.FileHandler("nn.log")
fh.setLevel(logging.INFO)



def find_closest_stations(distance_matrix, station_name, num_stations=3):
    distances = distance_matrix[station_name].sort_values()
    closest_stations = distances.index[1 : num_stations + 1]
    return closest_stations

# Next improvement would be to find the closest three stations to the one we are predicting and use their data
def nearest_time_point(series, time_point):
    nearest_index = abs(series - time_point).idxmin()
    return series.loc[nearest_index]

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
        X[f"attr_y{timestamp}_{station_name}"] = merged_data[station_name]

    return X

def load_and_preprocess_data(
    train_set_path: str, test_set_path: str, metadata_path: str
):
    metadata = pd.read_csv(metadata_path, parse_dates=["timestamp"])
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

        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # X_train = X_train.merge(train_set)
        # X_test = X_test.merge(test_set)

        X_train = get_historic_data(st_train, X_train, lookback=3)
        X_test = get_historic_data(st_test, X_test, lookback=3)
        timestamp_test_df = X_test["timestamp"].copy()
        #X_test.drop(["timestamp"], axis=1, inplace=True)
        #X_train.drop(["timestamp"], axis=1, inplace=True)

        # One hot encode the hour attribute
        # Convert columns to categorical data type with specified categories

        # X_train, X_test = standardize(X_train, X_test)
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # Count the nan values
        # print(X_train.isna().sum())
        # print(X_test.isna().sum())

        stations.append((st_train, timestamp_test_df, X_train, X_test, y_train, y_test))

    X_train_all = pd.concat([data[2] for data in stations], axis=1)
    y_train_all = pd.concat([data[4] for data in stations], axis=1)
    X_test_all = pd.concat([data[3] for data in stations], axis=1)
    y_test_all = pd.concat([data[5] for data in stations], axis=1)
    # Delete all the timestamps columns, but keep one
    timestamps = X_train_all.filter(regex='timestamp')
    X_train_all.drop(columns=timestamps.columns, inplace=True)
    X_train_all['timestamp'] = timestamps.iloc[:, 0]

    timestamps = X_test_all.filter(regex='timestamp')
    X_test_all.drop(columns=timestamps.columns, inplace=True)
    X_test_all['timestamp'] = timestamps.iloc[:, 0]

    # Convert the timestamp column to datetime
    X_train_all["timestamp"] = pd.to_datetime(
        X_train_all["timestamp"], format="%Y-%m-%d %H:%M:%S"
    )

    X_train_all = pd.merge(X_train_all, metadata, on="timestamp", how="left")
    X_test_all = pd.merge(X_test_all, metadata, on="timestamp", how="left")

    # drop the timestamp column
    X_train_all.drop(["timestamp"], axis=1, inplace=True)
    X_test_all.drop(["timestamp"], axis=1, inplace=True)


    #X_train_all["attr_hour"] = pd.Categorical(
    #    X_train_all["attr_hour"], categories=range(24)
    #)
    #X_test_all["attr_hour"] = pd.Categorical(X_test["attr_hour"], categories=range(24))

    #X_train_all["attr_minute"] = pd.Categorical(
    #    X_train_all["attr_minute"], categories=range(60)
    #)
    #X_test_all["attr_minute"] = pd.Categorical(
    #    X_test_all["attr_minute"], categories=range(60)
    #)

    #X_train_all["attr_weekday"] = pd.Categorical(
    #    X_train_all["attr_weekday"], categories=range(7)
    #)
    #X_test_all["attr_weekday"] = pd.Categorical(
    #    X_test_all["attr_weekday"], categories=range(7)
    #)

    ## Apply one-hot encoding
    #X_train_all = pd.get_dummies(
    #    X_train_all, columns=["attr_hour", "attr_minute", "attr_weekday"]
    #)
    #X_test_all = pd.get_dummies(
    #    X_test_all, columns=["attr_hour", "attr_minute", "attr_weekday"]
    #)

    return X_train_all, y_train_all, X_test_all, y_test_all


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

    def fit(self, layers, lr, epochs, x, y, batch_size=128):
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.act_fn = nn.GELU()

        self.loss_history = []

        x_tensor = torch.Tensor(x.values).to(self.device)
        y_tensor = torch.Tensor(y.values).to(self.device)
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.loss_history.append(epoch_loss/len(dataloader))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.act_fn(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x_tensor = torch.Tensor(x.values).to(self.device)
            predictions = self(x_tensor)
        return predictions.cpu().numpy()

    def score(self, x, y):
        y_pred = self.predict(x)
        return r2_score(y, y_pred)

    def mean_absolute_error(self, x, y):
        y_pred = self.predict(x)
        return mean_absolute_error(y, y_pred)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()



import numpy as np

def generate_random_layers(min_layers, max_layers, min_size, max_size, num_samples):
    """
    Generate a list of random layer sizes for neural network.

    Args:
    min_layers (int): Minimum number of layers.
    max_layers (int): Maximum number of layers.
    min_size (int): Minimum size of each layer.
    max_size (int): Maximum size of each layer.
    num_samples (int): Number of layer configurations to generate.

    Returns:
    list of lists: Each list contains a randomly generated layer configuration.
    """
    np.random.seed(42) # for reproducibility, optional
    layer_samples = []

    for _ in range(num_samples):
        num_layers = np.random.randint(min_layers, max_layers+1)
        layers = list(np.random.randint(min_size, max_size+1, num_layers))
        layers.sort(reverse=True)
        layers.insert(0, 359)
        layers.append(83)
        layer_samples.append(layers)

    return layer_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    
    lays = generate_random_layers(2, 5, 10, 100, 10)


    args = parser.parse_args()

    if args.train:
        stations = load_and_preprocess_data(
                "../data/bicikelj_train_indices.csv",
                "../data/bicikelj_test_indices.csv",
                "../data/bicikelj_metadata.csv",
        )
        X_train_all, y_train_all, X_test_all, y_test_all = stations

        print("[+] Finished loading and preprocessing data")

        X_train_all.to_csv("../data/X_train_all.csv", index=False)
        y_train_all.to_csv("../data/y_train_all.csv", index=False)

        #in_dim = X_train_all.shape[1]

        #in_dim_1 = int(in_dim * 0.65)
        #in_dim_2 = int(in_dim * 0.45)
        #in_dim_3 = int(in_dim * 0.25)
        #in_dim_4 = int(in_dim * 0.15)
        #print("in_dim_2: ", in_dim_2)
        #print("in_dim_3: ", in_dim_3)
        #print("in_dim_4: ", in_dim_4)
        #print('in_dim: ', in_dim)

        ## Create numpy array of range 0.1 to 1.0 with step 0.1
        ## Multiply each element with in_dim
        #elements = np.arange(0.1, 1.0, 0.2)
        ## Sort elements in descending order
        #elements = np.sort(elements)[::-1]
        #dimensions = [ int(in_dim * i) for i in elements]
        #dimensions = dimensions[:-1]
        #dimensions.insert(0, in_dim)
        #dimensions.append(83)
        #dimensions[4] = 70

        #print(dimensions)
        #layers = generate_random_layers(min_layers=1, max_layers=3, min_size=70, max_size=250, num_samples=400)
        #for layer in layers:
        #    nn = NN()
        #    nn.fit(dimensions, 0.001, 100, X_train_all, y_train_all)
        #    score = nn.mean_absolute_error(X_test_all, y_test_all)
        #    # Log score and layers to file
        #    with open('results.txt', 'a') as f:
        #        f.write(f'{score}, {layer}\n')

        xboost = xgb.XGBRegressor()
        xboost.fit(X_train_all, y_train_all)
        score = xboost.score(X_test_all, y_test_all)
        print(score)


    elif args.test:
        stations = load_and_preprocess_data(
                "../data/bicikelj_train.csv",
                "../data/bicikelj_test.csv",
                "../data/bicikelj_metadata.csv",
        )

        X_train_all, y_train_all, X_test_all, y_test_all = stations

        print("[+] Finished loading and preprocessing data")

        X_train_all.to_csv("../data/X_train_all.csv", index=False)
        y_train_all.to_csv("../data/y_train_all.csv", index=False)

        in_dim = X_train_all.shape[1]

        in_dim_1 = int(in_dim * 0.65)
        in_dim_2 = int(in_dim * 0.45)
        in_dim_3 = int(in_dim * 0.25)

        print("in_dim_1: ", in_dim_1)
        print("in_dim_2: ", in_dim_2)
        print("in_dim_3: ", in_dim_3)

        in_dim_1 = int(in_dim * 0.65)
        in_dim_2 = int(in_dim * 0.45)
        in_dim_3 = int(in_dim * 0.25)
        in_dim_4 = int(in_dim * 0.15)

        print("in_dim_1: ", in_dim_1)
        print("in_dim_2: ", in_dim_2)
        print("in_dim_3: ", in_dim_3)
        print("in_dim_4: ", in_dim_4)

        #nn = NN([in_dim, 230, 200, 180, 130, 90, 83])
        #print("[+] Finished creating NN model")
        #nn.fit(X_train_all, y_train_all)
        #nn.plot_loss()
        #dimensions = [359, 93, 83]
        #nn = NN()
        #nn.fit(dimensions, 0.001, 600, X_train_all, y_train_all)
        ### Predict
        #y_pred = nn.predict(X_test_all)
        ## Save predictions as pandas dataframe and save it to csv file

        xboost = xgb.XGBRegressor()
        xboost.fit(X_train_all, y_train_all)
        y_pred = xboost.predict(X_test_all)
        y_pred_df = pd.DataFrame(y_pred, columns=y_test_all.columns)
        import time
        print(y_pred_df)

        test_dataframe = pd.read_csv("../data/bicikelj_test.csv")
        y_pred_df.insert(0, "timestamp", test_dataframe["timestamp"])
        y_pred_df.to_csv(f"../data/xregressor_{str(int(time.time()))}.csv", index=False)


if __name__ == "__main__":
    main()
