import json

theta = {"theta0": 0,
         "theta1": 0}


def prediction(theta0, theta1, x):
    ''' function to predict the price of the car based on its mileage '''

    return ((theta1 * x) + theta0)


def main():

    try:

        theta0 = theta["theta0"]
        theta1 = theta["theta1"]

        with open("theta.json", "r+") as file:
            try:
                data_json = json.load(file)

            except json.decoder.JSONDecodeError:
                exit("Error: Empty Json file.")

        theta0 = data_json["theta0"]
        theta1 = data_json["theta1"]

        result = input("What is the estimated price for a car with: ")
        try:

            new = float(result)
            if new < 0:
                raise ValueError

            predict = prediction(theta0, theta1, new)
            if predict < 0:
                exit("Error: Your car is no longer valuable.")

            predict_r = round(predict, 2)
            print(f"Estimated price for a car with {new}km is {predict_r}e.")

        except ValueError:
            exit("Error: Incorrect value.")

    except IOError:
        exit("Error: Please, train your dataset.")


if __name__ == "__main__":
    main()
