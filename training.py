import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


def load_csv_data(path: str) -> pd.DataFrame:
    ''' function to load data from csv file '''

    if not path.endswith('.csv'):
        exit("Error: Not CSV file.")

    try:
        data = pd.read_csv(path)

        if data.empty:
            exit("Error: CSV file is empty.")
        else:
            return (data)

    except pd.errors.EmptyDataError:
        exit("Error: CSV file is empty.")
    except pd.errors.ParserError:
        exit("Error: incorrect file.")
    except FileNotFoundError:
        exit("Error: CSV file not found.")


def contains_string(lst):
    ''' function to check if a list contains any strings '''

    return any(isinstance(i, str) for i in lst)


def get_data(data: pd.DataFrame):
    ''' function to get different category '''

    km = data.iloc[0:len(data), 0].dropna()
    price = data.iloc[0:len(data), 1].dropna()

    if contains_string(km):
        exit("Error: 'km' contains string values.")

    if contains_string(price):
        exit("Error: 'price' contains string values.")

    if len(km) != len(price):
        exit("Error: data length mismatch.")

    for i in range(len(km)):
        if km[i] < 0 or price[i] < 0:
            exit("Error: negative value in data.")

    return (km, price)


def normalize_data(data: pd.DataFrame):
    ''' function to normalize data '''

    normalized_data = (data - data.min()) / (data.max() - data.min())

    km = normalized_data.iloc[0:len(normalized_data), 0]
    price = normalized_data.iloc[0:len(normalized_data), 1]

    return (km, price)


def gradient_descent(X, Y):
    ''' function to obtain the theta of the linear regression line '''

    lr = 0.01
    theta0 = 0
    theta1 = 0
    iterations = 10000
    m = len(X)

    for i in range(iterations):

        predict = (theta1 * X) + theta0
        tmp_theta0 = theta0 - (lr * (2/m) * sum(predict - Y))
        tmp_theta1 = theta1 - (lr * (2/m) * sum((predict - Y) * X))
        theta0 = tmp_theta0
        theta1 = tmp_theta1

    return (theta0, theta1)


def create_graph(theta0_d, theta1_d, km, price):
    ''' function to create graph '''

    try:
        Y_pred = (theta1_d * np.array(km)) + theta0_d

        plt.figure(figsize=(8, 6))
        plt.scatter(km, price, marker='o', color='red', label='Real data')
        plt.plot(km, Y_pred, color='blue', label='Regression line')
        plt.xlabel("Milegage")
        plt.ylabel("Price")
        plt.title("Linear regression")
        plt.legend()
        plt.show()

    except KeyboardInterrupt:
        exit("Error: program end by user.")


def save_theta_json(theta0_d, theta1_d):
    ''' function to save denormalize theta in JSON file '''

    theta = {"theta0": theta0_d, "theta1": theta1_d}

    with open("theta.json", "w+") as file:
        try:
            theta = json.load(file)

        except json.decoder.JSONDecodeError:
            json.dump(theta, file)


def main():

    data = load_csv_data("data.csv")

    km, price = get_data(data)
    km_normalize, price_normalize = normalize_data(data)

    theta0, theta1 = gradient_descent(km_normalize, price_normalize)
    print(f"theta0: {theta0}, theta1: {theta1}")

    X_min = np.min(km)
    X_max = np.max(km)
    Y_min = np.min(price)
    Y_max = np.max(price)

    theta1_d = theta1 * (Y_max - Y_min) / (X_max - X_min)
    theta0_d = price.mean() - theta1_d * km.mean()

    print("Theta after denormalization:")
    print(f"theta0: {theta0_d}, theta1: {theta1_d}")

    create_graph(theta0_d, theta1_d, km, price)
    save_theta_json(theta0_d, theta1_d)


if __name__ == "__main__":
    main()
