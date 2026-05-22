import numpy as np


def make_data():
    rng = np.random.default_rng(42)
    X = np.linspace(1, 10, 40)
    y = 4.5 * X + 8 + rng.normal(0, 3, size=X.size)
    return X, y


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def batch_gradient_descent(X, y, learning_rate=0.01, epochs=2000):
    weight = 0.0
    bias = 0.0
    sample_count = len(X)

    for _ in range(epochs):
        predictions = weight * X + bias
        errors = predictions - y
        grad_w = (2 / sample_count) * np.dot(errors, X)
        grad_b = (2 / sample_count) * errors.sum()
        weight -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weight, bias


def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=40):
    weight = 0.0
    bias = 0.0
    rng = np.random.default_rng(42)

    for _ in range(epochs):
        for index in rng.permutation(len(X)):
            prediction = weight * X[index] + bias
            error = prediction - y[index]
            grad_w = 2 * error * X[index]
            grad_b = 2 * error
            weight -= learning_rate * grad_w
            bias -= learning_rate * grad_b

    return weight, bias


def mini_batch_gradient_descent(X, y, learning_rate=0.01, epochs=200, batch_size=8):
    weight = 0.0
    bias = 0.0
    rng = np.random.default_rng(42)

    for _ in range(epochs):
        indices = rng.permutation(len(X))
        shuffled_X = X[indices]
        shuffled_y = y[indices]

        for start in range(0, len(X), batch_size):
            stop = start + batch_size
            batch_X = shuffled_X[start:stop]
            batch_y = shuffled_y[start:stop]

            predictions = weight * batch_X + bias
            errors = predictions - batch_y
            grad_w = (2 / len(batch_X)) * np.dot(errors, batch_X)
            grad_b = (2 / len(batch_X)) * errors.sum()
            weight -= learning_rate * grad_w
            bias -= learning_rate * grad_b

    return weight, bias


if __name__ == "__main__":
    X, y = make_data()

    batch_w, batch_b = batch_gradient_descent(X, y)
    sgd_w, sgd_b = stochastic_gradient_descent(X, y)
    mini_w, mini_b = mini_batch_gradient_descent(X, y)

    print("Batch GD:", batch_w, batch_b, mean_squared_error(y, batch_w * X + batch_b))
    print("SGD:", sgd_w, sgd_b, mean_squared_error(y, sgd_w * X + sgd_b))
    print("Mini-batch GD:", mini_w, mini_b, mean_squared_error(y, mini_w * X + mini_b))

