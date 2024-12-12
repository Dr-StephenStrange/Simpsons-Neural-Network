import os
import numpy as np
import cv2
import gradio as gr
from tensorflow.keras.models import load_model
import tensorflow


# Render deployment
RENDER = True
port = int(os.environ.get("PORT", 7860))

# Configuration
MODEL_PATH = "model_dir"
TEST_DATA_DIR = "Dataset/testset"
IMAGE_SIZE = 64

CHARACTER_LABELS_MAP = {
    0: 'abraham_grampa_simpson',
    1: 'apu_nahasapeemapetilon',
    2: 'bart_simpson',
    3: 'charles_montgomery_burns',
    4: 'chief_wiggum',
    5: 'comic_book_guy',
    6: 'edna_krabappel',
    7: 'homer_simpson',
    8: 'kent_brockman',
    9: 'krusty_the_clown',
    10: 'lisa_simpson',
    11: 'marge_simpson',
    12: 'milhouse_van_houten',
    13: 'moe_szyslak',
    14: 'ned_flanders',
    15: 'nelson_muntz',
    16: 'principal_skinner',
    17: 'sideshow_bob'
}

CHAR_NAME_TO_LABEL = {v: k for k,v in CHARACTER_LABELS_MAP.items()}

# Load model
model = load_model(MODEL_PATH)

def get_testset_images(test_dir):
    """Get a list of available test images from the testset directory."""
    if not os.path.exists(test_dir):
        return []
    files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    return files

test_images = get_testset_images(TEST_DATA_DIR)

def preprocess_image(img):
    """Resize and normalize the image for model prediction."""
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
    img_resized = img_resized.astype('float32') / 255.0
    return img_resized

def parse_ground_truth_from_filename(filename):
    """Parse the character name from the filename and get the corresponding label."""
    base_name = "_".join(filename.split("_")[:-1])
    label = CHAR_NAME_TO_LABEL.get(base_name, None)
    return label

def show_test_image(source_type, selected_image):
    """
    If source_type is Testset and a selected_image is chosen, return that image for display.
    Otherwise, return None to clear the image.
    """
    if source_type == "Testset" and selected_image:
        img_path = os.path.join(TEST_DATA_DIR, selected_image)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img_rgb
    return None

def predict_image(source_type, selected_image, uploaded_image):
    """
    Predict the character in the image, show confidence.
    If source_type is 'Testset', we know ground truth and can show correctness.
    """
    if source_type == "Testset":
        if not selected_image:
            return "No image selected.", "", ""
        img_path = os.path.join(TEST_DATA_DIR, selected_image)
        img = cv2.imread(img_path)
        if img is None:
            return f"Could not load {selected_image}", "", ""
        input_img = preprocess_image(img)
        input_img = np.expand_dims(input_img, axis=0)

        predictions = model.predict(input_img)
        pred_label = np.argmax(predictions[0])
        pred_name = CHARACTER_LABELS_MAP[pred_label]
        confidence = predictions[0][pred_label] * 100

        # Ground truth parsing
        gt_label = parse_ground_truth_from_filename(selected_image)
        if gt_label is not None:
            gt_name = CHARACTER_LABELS_MAP[gt_label]
            # Check correctness
            if gt_label == pred_label:
                correctness = f"<span style='color:green'>Correct</span>"
            else:
                correctness = f"<span style='color:red'>Incorrect (GT: {gt_name})</span>"
        else:
            correctness = ""  # If we cannot parse ground truth

        return f"Predicted: {pred_name}", f"Confidence: {confidence:.2f}%", correctness

    elif source_type == "Upload":
        if uploaded_image is None:
            return "No image uploaded.", "", ""

        # uploaded_image is a numpy array (H,W,C) from Gradio (RGB format)
        img = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)
        input_img = preprocess_image(img)
        input_img = np.expand_dims(input_img, axis=0)

        predictions = model.predict(input_img)
        pred_label = np.argmax(predictions[0])
        pred_name = CHARACTER_LABELS_MAP[pred_label]
        confidence = predictions[0][pred_label] * 100

        # No ground truth for uploaded images
        return f"Predicted: {pred_name}", f"Confidence: {confidence:.2f}%", ""

    else:
        return "Invalid source type", "", ""


# Build the gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Simpsons Character Classifier")

    source_type = gr.Radio(["Testset", "Upload"], label="Image Source", value="Testset")

    with gr.Row():
        selected_image = gr.Dropdown(choices=test_images, label="Select image from testset")
        uploaded_image = gr.Image(label="Upload your own image")

    # Add an image component to display the currently selected testset image (if any)
    display_image = gr.Image(label="Selected Testset Image", interactive=False)

    predict_button = gr.Button("Predict")

    output_pred = gr.Markdown()
    output_conf = gr.Markdown()
    output_correctness = gr.Markdown()

    def on_source_change(source):
        # If source is Testset, enable selected_image and disable upload
        # If source is Upload, enable upload and disable selected_image
        if source == "Testset":
            return gr.update(interactive=True), gr.update(interactive=False), None
        else:
            return gr.update(interactive=False), gr.update(interactive=True), None

    # Update image display and states when source_type changes
    source_type.change(
        on_source_change,
        inputs=source_type,
        outputs=[selected_image, uploaded_image, display_image]
    )

    # When selected_image changes, show that image if source_type is Testset
    selected_image.change(
        show_test_image,
        inputs=[source_type, selected_image],
        outputs=display_image
    )

    predict_button.click(
        fn=predict_image,
        inputs=[source_type, selected_image, uploaded_image],
        outputs=[output_pred, output_conf, output_correctness]
    )

if RENDER:
    demo.launch(server_name="0.0.0.0", server_port=port)
else:
    demo.launch()
