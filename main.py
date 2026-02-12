def test_config():
    import pandas as pd
    print(pd.__version__)
    import keras
    print(keras.backend.backend())

def number_recognition_model():
    import pandas as pd
    train = pd.read_csv("plate_recognition/emnist-digits-train.csv",sep=",", nrows=20)