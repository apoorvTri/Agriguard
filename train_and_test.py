# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # -----------------------------
# # BASIC CONFIG
# # -----------------------------
# IMG_SIZE = (128, 128)
# BATCH_SIZE = 32
# EPOCHS = 5   # increase later

# TRAIN_DIR = "train"
# VAL_DIR = "valid"
# TEST_IMAGE_PATH = "test_images/test1.jpg"

# # -----------------------------
# # LOAD DATASET
# # -----------------------------
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     TRAIN_DIR,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="categorical"
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     VAL_DIR,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="categorical"
# )

# class_names = train_ds.class_names
# num_classes = len(class_names)

# print("\nNumber of Classes:", num_classes)
# print("Class Names:", class_names)

# # Normalize images
# normalization = tf.keras.layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization(x), y))

# # -----------------------------
# # BUILD CNN MODEL
# # -----------------------------
# model = Sequential([
#     Conv2D(32, 3, activation="relu", input_shape=(128,128,3)),
#     MaxPooling2D(),

#     Conv2D(64, 3, activation="relu"),
#     MaxPooling2D(),

#     Conv2D(128, 3, activation="relu"),
#     MaxPooling2D(),

#     Flatten(),
#     Dense(256, activation="relu"),
#     Dropout(0.3),
#     Dense(num_classes, activation="softmax")
# ])

# model.compile(
#     optimizer="adam",
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # TRAIN MODEL
# # -----------------------------
# print("\n🚀 Training Started...\n")
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS
# )

# # -----------------------------
# # EVALUATE MODEL
# # -----------------------------
# train_loss, train_acc = model.evaluate(train_ds, verbose=0)
# val_loss, val_acc = model.evaluate(val_ds, verbose=0)

# print("\n📊 FINAL RESULTS")
# print(f"Training Accuracy   : {train_acc*100:.2f}%")
# print(f"Validation Accuracy : {val_acc*100:.2f}%")

# # -----------------------------
# # ACCURACY & LOSS PLOTS
# # -----------------------------
# epochs_range = range(1, EPOCHS + 1)

# plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# plt.plot(epochs_range, history.history['accuracy'], label="Training Accuracy")
# plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
# plt.title("Accuracy vs Epochs")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(epochs_range, history.history['loss'], label="Training Loss")
# plt.plot(epochs_range, history.history['val_loss'], label="Validation Loss")
# plt.title("Loss vs Epochs")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

# plt.tight_layout()
# plt.show()

# # -----------------------------
# # SAVE MODEL
# # -----------------------------
# model.save("trained_model.h5")
# print("\n✅ Model saved as trained_model.h5")

# # -----------------------------
# # TEST WITH ONE IMAGE
# # -----------------------------
# img = load_img(TEST_IMAGE_PATH, target_size=IMG_SIZE)
# img_array = img_to_array(img) / 255.0
# img_array = np.expand_dims(img_array, axis=0)

# prediction = model.predict(img_array)
# predicted_class = class_names[np.argmax(prediction)]
# confidence = np.max(prediction)

# print("\n🖼 TEST IMAGE RESULT")
# print("Predicted Class:", predicted_class)
# print(f"Confidence: {confidence*100:.2f}%")

# plt.imshow(img)
# plt.title(f"{predicted_class} ({confidence*100:.2f}%)")
# plt.axis("off")
# plt.show()


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

TRAIN_DIR = "train"
VAL_DIR = "valid"
TEST_IMAGE_PATH = "test/AppleCedarRust2.jpg"
MODEL_PATH = "trained_model.h5"

# -----------------------------
# LOAD DATA (for class names)
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Normalize
norm = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (norm(x), y))
val_ds = val_ds.map(lambda x, y: (norm(x), y))

# -----------------------------
# TRAIN ONLY IF MODEL NOT FOUND
# -----------------------------
if os.path.exists(MODEL_PATH):
    print("\n✅ Trained model found. Skipping training & evaluation.")
    model = load_model(MODEL_PATH)

else:
    print("\n🚀 Training started...")

    model = Sequential([
        Conv2D(32, 3, activation="relu", input_shape=(128,128,3)),
        MaxPooling2D(),

        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(),

        Conv2D(128, 3, activation="relu"),
        MaxPooling2D(),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # -----------------------------
    # EVALUATION (ONLY AFTER TRAINING)
    # -----------------------------
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)

    print("\n📊 TRAINING RESULTS")
    print(f"Training Accuracy   : {train_acc*100:.2f}%")
    print(f"Validation Accuracy : {val_acc*100:.2f}%")

    # Accuracy graph (optional but nice)
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()

    model.save(MODEL_PATH)
    print("\n✅ Model trained and saved.")

# -----------------------------
# TESTING (ALWAYS RUNS)
# -----------------------------
img = load_img(TEST_IMAGE_PATH, target_size=IMG_SIZE)
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("\n🖼 TEST IMAGE RESULT")
print("Predicted Class:", predicted_class)
print(f"Confidence: {confidence*100:.2f}%")

plt.imshow(img)
plt.title(f"{predicted_class} ({confidence*100:.2f}%)")
plt.axis("off")
plt.show()