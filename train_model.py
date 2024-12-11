import os
import sys
import itertools
import logging
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# =============================================================================
# Configuration
# =============================================================================
TRAIN_DATA_DIR = "Dataset/simpsons_dataset"
TEST_DATA_DIR = "Dataset/testset"
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
DECAY = 1e-6
ROTATION_RANGE = 10
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1

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
NUM_CLASSES = len(CHARACTER_LABELS_MAP)

if len(CHARACTER_LABELS_MAP) != NUM_CLASSES:
    logger.error("NUM_CLASSES does not match the number of entries in CHARACTER_LABELS_MAP.")
    sys.exit(1)


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_training_data(data_dir: str, labels_map: dict, images_per_class: int = None) -> (np.ndarray, np.ndarray):
    X_train, Y_train = [], []
    if not os.path.exists(data_dir):
        logger.error(f"Training directory {data_dir} does not exist.")
        return np.array([]), np.array([])

    for label, character_name in labels_map.items():
        character_dir = os.path.join(data_dir, character_name)
        if not os.path.exists(character_dir):
            logger.warning(f"Directory for character '{character_name}' not found. Skipping this class.")
            continue

        image_files = os.listdir(character_dir)
        if images_per_class is not None:
            image_files = image_files[:images_per_class]

        for img_file in image_files:
            img_path = os.path.join(character_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}. Skipping.")
                continue
            img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
            X_train.append(img_resized)
            Y_train.append(label)

    return np.array(X_train), np.array(Y_train)


def load_test_data(data_dir: str, labels_map: dict) -> (np.ndarray, np.ndarray):
    X_test, Y_test = [], []
    if not os.path.exists(data_dir):
        logger.error(f"Test directory {data_dir} does not exist.")
        return np.array([]), np.array([])

    for img_file in os.listdir(data_dir):
        base_name = "_".join(img_file.split("_")[:-1])
        label = next((l for l, c in labels_map.items() if c == base_name), None)
        if label is None:
            continue

        img_path = os.path.join(data_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Could not read image: {img_path}. Skipping.")
            continue
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        X_test.append(img_resized)
        Y_test.append(label)

    return np.array(X_test), np.array(Y_test)


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_confusion_matrix(cm: np.ndarray, classes: list, normalize: bool = False,
                          title: str = 'Confusion Matrix', cmap=plt.cm.Blues):
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


# =============================================================================
# Model Definition
# =============================================================================
def build_convolutional_model(input_shape: tuple, num_classes: int) -> Sequential:
    model = Sequential()

    # First block
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Second block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Third block
    model.add(Conv2D(86, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(86, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    optimizer = RMSprop(learning_rate=LEARNING_RATE, decay=DECAY)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate_model(X_train_full: np.ndarray, Y_train_full: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, labels_map: dict):
    if X_train_full.size == 0 or Y_train_full.size == 0:
        logger.error("Training data is empty. Cannot train the model.")
        return

    if X_test.size == 0 or Y_test.size == 0:
        logger.error("Test data is empty. Cannot evaluate the model.")
        return

    # Split training data into train and val sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=42
    )

    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    input_shape = (img_rows, img_cols, 3)
    num_classes = Y_train.shape[1]

    model = build_convolutional_model(input_shape, num_classes)

    # Create separate generators for training and validation
    train_datagen = ImageDataGenerator(
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        horizontal_flip=True
    )
    train_datagen.fit(X_train)

    val_datagen = ImageDataGenerator()  # Typically no augmentation for validation
    val_datagen.fit(X_val)

    train_generator = train_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(X_val, Y_val, batch_size=BATCH_SIZE)

    # Callbacks: EarlyStopping and ReduceLROnPlateau
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
        mode='max'
    )

    logger.info("Starting model training with early stopping and LR reduction...")

    model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=val_generator,
        validation_steps=len(X_val) // BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr]
    )

    logger.info("Evaluating model on test data...")
    score = model.evaluate(X_test, Y_test, verbose=0)
    logger.info(f"Test Accuracy: {score[1]:.4f}")

    y_pred = model.predict(X_test)
    logger.info("Classification Report:")
    print(classification_report(np.argmax(Y_test, axis=1), np.argmax(y_pred, axis=1), target_names=list(labels_map.values())))

    Y_pred_classes = np.argmax(y_pred, axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plot_confusion_matrix(confusion_mtx, classes=list(labels_map.values()))
    plt.show()

    model.save("model.keras")
    logger.info("Model saved as 'model.keras'.")


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate a CNN model on The Simpsons Dataset")
    parser.add_argument("--images_per_class", type=int, default=None,
                        help="Limit number of images loaded per class for debugging purposes.")
    args = parser.parse_args()

    logger.info("Loading training data...")
    X_train, Y_train = load_training_data(TRAIN_DATA_DIR, CHARACTER_LABELS_MAP, images_per_class=args.images_per_class)

    logger.info("Loading testing data...")
    X_test, Y_test = load_test_data(TEST_DATA_DIR, CHARACTER_LABELS_MAP)

    # Normalize data
    if X_train.size > 0:
        X_train = X_train.astype('float32') / 255.0
    if X_test.size > 0:
        X_test = X_test.astype('float32') / 255.0

    if Y_train.size > 0:
        Y_train = to_categorical(Y_train, num_classes=NUM_CLASSES)
    if Y_test.size > 0:
        Y_test = to_categorical(Y_test, num_classes=NUM_CLASSES)

    logger.info(f"Training Data Shape: {X_train.shape}, {Y_train.shape}")
    logger.info(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}")

    train_and_evaluate_model(X_train, Y_train, X_test, Y_test, CHARACTER_LABELS_MAP)


if __name__ == "__main__":
    main()
