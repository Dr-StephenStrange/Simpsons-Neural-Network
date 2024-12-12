from keras.models import load_model
from keras.utils import plot_model
import os

import pydot
import pydotplus
import graphviz

for path in os.environ["PATH"].split(";"):
    print(path)

os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

# Load the SavedModel
model = load_model("model.keras")

# Save the architecture as an image
plot_model(model, to_file='cnn_architecture.png', show_shapes=True, show_layer_names=True)
