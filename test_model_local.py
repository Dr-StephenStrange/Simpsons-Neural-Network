import os
import numpy as np
import cv2
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# Configuration
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    Plot confusion matrix.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def load_and_preprocess_test_data(test_dir, labels_map):
    """
    Load all images from the test directory, extract the label from the filename,
    and return arrays of processed images, original images, and labels.

    Returns:
        X_test (np.ndarray): processed images (resized and normalized)
        X_test_original (list): original images in their original resolution and color
        Y_test (np.ndarray): labels
    """
    X_test = []
    Y_test = []
    X_test_original = []

    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist.")
        return np.array([]), [], np.array([])

    for img_file in os.listdir(test_dir):
        # Extract character name from file name
        base_name = "_".join(img_file.split("_")[:-1])
        label = None
        for l, c in labels_map.items():
            if c == base_name:
                label = l
                break
        if label is None:
            # If we cannot find the class, skip this image
            continue

        img_path = os.path.join(test_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}. Skipping.")
            continue

        # Store the original image
        X_test_original.append(img)

        # Process for prediction
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        img_resized = img_resized.astype('float32') / 255.0  # Normalize for model
        X_test.append(img_resized)
        Y_test.append(label)

    return np.array(X_test), X_test_original, np.array(Y_test)


def main():
    # Load the saved model
    model = load_model("model.keras")
    print("Model loaded successfully.")

    # Load and preprocess test data
    X_test, X_test_original, Y_test = load_and_preprocess_test_data(TEST_DATA_DIR, CHARACTER_LABELS_MAP)

    if X_test.size == 0:
        print("No test data found. Exiting.")
        return

    # Predict
    predictions = model.predict(X_test)
    Y_pred = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = np.mean(Y_pred == Y_test)
    print(f"Test Accuracy on all images in {TEST_DATA_DIR}: {accuracy:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred, target_names=list(CHARACTER_LABELS_MAP.values())))

    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    plot_confusion_matrix(cm, classes=list(CHARACTER_LABELS_MAP.values()))
    plt.show()

    # Display a 4x4 grid of random sample predictions (raw images)
    num_images = min(len(X_test), 16)
    selected_indices = np.random.choice(len(X_test), num_images, replace=False)

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(selected_indices):
        plt.subplot(4, 4, i+1)
        # Display the original, unprocessed image
        img_disp = X_test_original[idx]
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        predicted_label = Y_pred[idx]
        predicted_name = CHARACTER_LABELS_MAP[predicted_label]
        confidence = predictions[idx, predicted_label] * 100
        plt.title(f"{predicted_name}\n{confidence:.2f}%")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
