from tensorflow.keras.models import load_model


MODEL_PATH = "model.keras"
model = load_model(MODEL_PATH)

model.export("model_dir")  # Save as a TensorFlow SavedModel

