import tensorflow as tf

# Path to your SavedModel directory
MODEL_PATH = "model_dir"

# Load the SavedModel
model = tf.saved_model.load(MODEL_PATH)

# Print the available signatures
print("Model Signatures:")
for signature_key in model.signatures.keys():
    print(f" - {signature_key}")

# Inspect the "serving_default" signature (commonly used for inference)
if "serving_default" in model.signatures:
    serving_fn = model.signatures["serving_default"]

    # Print inputs
    print("\nInputs:")
    for input_key, input_tensor in serving_fn.structured_input_signature[1].items():
        print(f" - {input_key}: {input_tensor}")

    # Print outputs
    print("\nOutputs:")
    for output_key, output_tensor in serving_fn.structured_outputs.items():
        print(f" - {output_key}: {output_tensor}")
else:
    print("No 'serving_default' signature found in the model.")
