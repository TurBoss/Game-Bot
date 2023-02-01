# Arda Mavi
import numpy as np
from PIL import Image


def predict(model, X):
    X = Image.fromarray(X.astype('uint8'))
    X = X.resize((150, 150))
    X = np.array(X).astype('float32') / 255.
    Y = model.predict(X.reshape(1, 150, 150, 3))
    return Y
