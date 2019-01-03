from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from psokeras import Optimizer

N = 30 # number of particles
STEPS = 300 # number of steps
LOSS = 'mse' # Loss function
BATCH_SIZE = 32 # Size of batches to train on


def build_model(loss):
    model = Sequential()
    model.add(Dense(4, activation='sigmoid', input_dim=4, use_bias=True))
    model.add(Dense(3, activation='softmax', use_bias=True))

    model.compile(loss=loss,
                  optimizer='adam')

    return model


def vanilla_backpropagation():
    # Backprop
    best_model = None
    best_score = 100.0

    for i in range(N):
        model_s = build_model(LOSS)
        model_s.fit(x_train, y_train,
                    epochs=STEPS,
                    batch_size=BATCH_SIZE,
                    verbose=0)
        train_score = model_s.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=0)
        if train_score < best_score:
            best_model = model_s
            best_score = train_score
    return best_model


if __name__ == "__main__":
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data,
                                                        keras.utils.to_categorical(iris.target, num_classes=None),
                                                        test_size = 0.5,
                                                        random_state = 0,
                                                        stratify = iris.target)

    # Backpropagation
    model_s = vanilla_backpropagation()
    b_train_score = model_s.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    b_test_score = model_s.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    print "Backprop -- train: {:.4f}  test: {:.4f}".format(b_train_score, b_test_score)

    # PSO
    model_p = build_model(LOSS)
    pso = Optimizer(model_p,LOSS,
                    n=N,
                    acceleration=1.0,
                    local_rate=0.6,
                    global_rate=0.4
                    )

    pso.fit(x_train, y_train, steps=STEPS, batch_size=BATCH_SIZE)
    model_p = pso.get_best_model()

    p_train_score = model_p.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    p_test_score = model_p.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    print "PSO -- train: {:.4f}  test: {:.4f}".format(p_train_score, p_test_score)
