import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

data = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/potato deases prediction/Training/PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
    label_mode='int'
)

from google.colab import drive
drive.mount('/content/drive')

class_names = data.class_names
n_classes = len(class_names)

for image_batch, label_batch in data.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))
    plt.title(class_names[label_batch[0]])

train_size = 0.8
len(data)*train_size

train_ds = data.take(54)

test_ds = data.skip(54)

val_size = 0.1
val_ds = test_ds.take(6)

test_ds = test_ds.skip(6)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Building the Model
## Creating a Layer for Resizing and Normalization

resize_and_rescale=tf.keras.Sequential([
    layers.Resizing(256,256),
    layers.Rescaling(1.0/255, input_shape=(256, 256, 3)),
])

data_augmentation=tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Reshape data for Random Forest
def preprocess_for_rf(dataset):
  images = []
  labels = []
  for image_batch, label_batch in dataset:
    for image, label in zip(image_batch, label_batch):
      images.append(image.numpy().flatten())  # Flatten image
      labels.append(label.numpy())
  return np.array(images), np.array(labels)

X_train, y_train = preprocess_for_rf(train_ds)
X_val, y_val = preprocess_for_rf(val_ds)
X_test, y_test = preprocess_for_rf(test_ds)
# Create and train the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")


import joblib

# Save the model to a file
joblib.dump(rf_model, 'Random_forest.pkl')
print("Model saved as Random_forest.pkl")

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Reshape data for KNN
def preprocess_for_knn(dataset):
  images = []
  labels = []
  for image_batch, label_batch in dataset:
    for image, label in zip(image_batch, label_batch):
      images.append(image.numpy().flatten())  # Flatten image
      labels.append(label.numpy())
  return np.array(images), np.array(labels)

X_train, y_train = preprocess_for_knn(train_ds)
X_val, y_val = preprocess_for_knn(val_ds)
X_test, y_test = preprocess_for_knn(test_ds)
# Create and train the KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# Predict on the test set
y_pred = knn_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy}")

# Save the model to a file
joblib.dump(knn_model, 'knn_model.pkl')
print("Model saved as knn_model.pkl")

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reshape data for Naive Bayes
def preprocess_for_nb(dataset):
  images = []
  labels = []
  for image_batch, label_batch in dataset:
    for image, label in zip(image_batch, label_batch):
      images.append(image.numpy().flatten())  # Flatten image
      labels.append(label.numpy())
  return np.array(images), np.array(labels)

X_train, y_train = preprocess_for_nb(train_ds)
X_val, y_val = preprocess_for_nb(val_ds)
X_test, y_test = preprocess_for_nb(test_ds)

# Create and train the Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Accuracy: {accuracy}")

# Save the model (optional)
import joblib
joblib.dump(nb_model, 'naive_bayes_model.pkl')
print("Model saved as naive_bayes_model.pkl")

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Reshape data for SVM
def preprocess_for_svm(dataset):
  images = []
  labels = []
  for image_batch, label_batch in dataset:
    for image, label in zip(image_batch, label_batch):
      images.append(image.numpy().flatten())  # Flatten image
      labels.append(label.numpy())
  return np.array(images), np.array(labels)

X_train, y_train = preprocess_for_svm(train_ds)
X_val, y_val = preprocess_for_svm(val_ds)
X_test, y_test = preprocess_for_svm(test_ds)

# Create and train the SVM classifier (using OvR or OvO automatically)
svm_model = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')  # Explicitly set 'ovr' if needed
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy}")

# Save the model (optional)
import joblib
joblib.dump(svm_model, 'svm_model.pkl')
print("Model saved as svm_model.pkl")


import tensorflow as tf
from tensorflow.keras import models, layers


# Define AlexNet architecture
model = models.Sequential([
    layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)

scores = model.evaluate(test_ds)

scores

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# 1. Load the pre-trained EfficientNetB7 model:
base_model = EfficientNetB7(weights='imagenet', include_top=False,
                           input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

# 2. Freeze the base model layers:
base_model.trainable = False

# 3. Add new classification layers on top:
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(n_classes, activation='softmax')(x)

# 4. Create the final model:
model = Model(inputs=base_model.input, outputs=predictions)

# 5. Compile the model:
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 6. Train the model:
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)

scores = model.evaluate(test_ds)
scores
model.save('efficientnetb7.h5')
print("Model saved as efficientnetb7.h5")

import matplotlib.pyplot as plt


# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show() 
