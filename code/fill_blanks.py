import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

ljubljana_lat = 46.0569
ljubljana_lon = 14.5058


# https://archive-api.open-meteo.com/v1/archive?latitude=46.05&longitude=14.51&start_date=2022-05-03&end_date=2022-09-14&hourly=
def get_day_weater(d, m, y, lat, lon):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = f"{y}-{m:02d}-{d-1:02d}"
    end_date = f"{y}-{m:02d}-{d:02d}"
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    end_date = int(end_date.timestamp())
    start_date = end_date - 24 * 60 * 60
    start_date = datetime.fromtimestamp(start_date).strftime("%Y-%m-%d")
    end_date = datetime.fromtimestamp(end_date).strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "relativehumidity_2m,surface_pressure,shortwave_radiation,windspeed_10m",
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
    else:
        print(f"{start_date} - {end_date}: {response.status_code} OK")

    time.sleep(1)

    return response.json()


def gen_weather_attr(df):
    rows = [
        "attr_relative_humidity",
        "attr_surface_pressure",
        "attr_shortwave_radiation",
        "attr_windspeed",
    ]
    for i in range(len(rows)):
        for j in range(4):
            df[f"attr_{rows[i]}_{j}h_before"] = np.nan


def apply_weather_data(df):
    gen_weather_attr(df)
    d, m, y = 0, 0, 0
    TODAY = 24
    weather_data = None
    for ix, row in df.iterrows():
        if (
            row["timestamp"].day != d
            or row["timestamp"].month != m
            or row["timestamp"].year != y
        ):
            d, m, y = (
                row["timestamp"].day,
                row["timestamp"].month,
                row["timestamp"].year,
            )
            weather_data = get_day_weater(d, m, y, ljubljana_lat, ljubljana_lon)
        rows = [
            "attr_relative_humidity",
            "attr_surface_pressure",
            "attr_shortwave_radiation",
            "attr_windspeed",
        ]
        rows_json = [
            "relativehumidity_2m",
            "surface_pressure",
            "shortwave_radiation",
            "windspeed_10m",
        ]
        CURRENT_HOUR = row["timestamp"].hour + TODAY
        for i in range(len(rows)):
            for j in range(4):
                df.at[ix, f"attr_{rows[i]}_{j}h_before"] = weather_data["hourly"][
                    f"{rows_json[i]}"
                ][CURRENT_HOUR - j]

    df = df.dropna()
    print(df)
    return df


def add_features(df):
    timestamps = df["timestamp"]
    features_dataframe = pd.DataFrame()
    features_dataframe["attr_day"] = pd.to_datetime(timestamps).dt.day
    features_dataframe["attr_month"] = pd.to_datetime(timestamps).dt.month
    features_dataframe["attr_hour"] = pd.to_datetime(timestamps).dt.hour
    features_dataframe["attr_minute"] = pd.to_datetime(timestamps).dt.minute
    features_dataframe["attr_weekday"] = pd.to_datetime(timestamps).dt.weekday
    features_dataframe["attr_weekend"] = features_dataframe["attr_weekday"].apply(
        lambda x: 1 if x >= 5 else 0
    )

    features_dataframe["attr_day_of_year"] = pd.to_datetime(timestamps).dt.dayofyear

    features_dataframe["rush_hour"] = features_dataframe["attr_hour"].apply(
        lambda x: 1 if (x >= 7 and x <= 9) or (x >= 16 and x <= 18) else 0
    )

    features_dataframe["attr_daylight"] = features_dataframe["attr_hour"].apply(
        lambda x: 1 if (x >= 6 and x <= 21) else 0
    )

    features_dataframe["attr_business_hours"] = features_dataframe["attr_hour"].apply(
        lambda x: 1 if (x >= 9 and x <= 17) else 0
    )
    features_dataframe["attr_night_hours"] = features_dataframe["attr_hour"].apply(
        lambda x: 1 if (x >= 22 or x <= 5) else 0
    )

    return df.join(features_dataframe)


def generate_metadata(df):
    metadata_df = pd.DataFrame(df.copy(), columns=["timestamp"])
    metadata_df = add_features(metadata_df)
    metadata_df = apply_weather_data(metadata_df)
    metadata_df.to_csv("../data/bicikelj_metadata.csv", index=False)


def generate_train_test_data(train_data, test_data):
    # Join the train and test datestes
    all_data = pd.concat([train_data, test_data])
    all_data.sort_index(inplace=True)
    all_data.to_csv("../data/bicikelj_all.csv")
    all_data = all_data.reset_index().rename(columns={"index": "timestamp"})

    generate_metadata(all_data)

    nan_indices = all_data[all_data.isna().any(axis=1)].index

    test_indices = nan_indices.copy() - 2
    test_indices = all_data.iloc[test_indices].copy()
    train_indices = all_data.drop(test_indices.index).copy()

    train_indices.dropna(inplace=True)
    test_indices.dropna(inplace=True)
    train_indices.to_csv("../data/bicikelj_train_indices.csv", index=False)
    test_indices.to_csv("../data/bicikelj_test_indices.csv", index=False)

    # Read the metadata
    metadata = pd.read_csv("../data/bicikelj_metadata.csv", parse_dates=["timestamp"])
    print(metadata)


def generate_prediction_data(train_data, test_data):
    # Join the train and test datestes
    all_data = pd.concat([train_data, test_data])
    all_data.sort_index(inplace=True)
    all_data = all_data.reset_index().rename(columns={"index": "timestamp"})

    # Read the metadata
    metadata = pd.read_csv("../data/bicikelj_metadata.csv", parse_dates=["timestamp"])


def main():
    train_data = pd.read_csv(
        "../data/bicikelj_train.csv", parse_dates=["timestamp"], index_col=["timestamp"]
    )
    test_data = pd.read_csv(
        "../data/bicikelj_test.csv", parse_dates=["timestamp"], index_col=["timestamp"]
    )

    generate_train_test_data(train_data, test_data)


if __name__ == "__main__":
    main()
