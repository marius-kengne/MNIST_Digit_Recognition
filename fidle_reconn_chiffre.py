import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import sys
from importlib import reload

# Ajouter le chemin parent pour accéder au module fidle.pwk
sys.path.append('..')

# Charger le dataset MNIST depuis TensorFlow
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Redimensionner les données
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("x_train : ", x_train.shape)
print("y_train : ", y_train.shape)
print("x_test : ", x_test.shape)
print("y_test : ", y_test.shape)

print("Before normalization : Min={}, Max={}".format(x_train.min(), x_train.max()))

# Normaliser les données
xmax = x_train.max()
x_train = x_train / xmax
x_test = x_test / xmax

print("After normalization : Min={}, Max={}".format(x_train.min(), x_train.max()))

print("Have a look of images")

# Définir la fonction plot_images
def plot_images(images, labels, indices, x_size=5, colorbar=False, saveas=None):
    plt.figure(figsize=(x_size, x_size))
    for i, idx in enumerate(indices):
        plt.subplot(1, len(indices), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[idx].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(labels[idx])
        if colorbar:
            plt.colorbar()
    if saveas:
        plt.savefig(saveas)
    plt.show()

# Définir la fonction plot_multiple_images
def plot_multiple_images(images, labels, indices, columns=12, saveas=None):
    rows = (len(indices) + columns - 1) // columns
    plt.figure(figsize=(columns, rows))
    for i, idx in enumerate(indices):
        plt.subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[idx].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(labels[idx])
    if saveas:
        plt.savefig(saveas)
    plt.show()

# Afficher une seule image
#plot_images(x_train, y_train, [27], x_size=5, colorbar=True, saveas='01-one-digit')

# Afficher plusieurs images
#plot_multiple_images(x_train, y_train, range(5, 41), columns=12, saveas='02-many-digits')




model=keras.models.Sequential()

model.add(keras.layers.Input((28,28,1)))

model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


print("Training the model")

batch_size = 512
epochs = 16

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)


print("Evaluating the model")

score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss         : , {score[0]:4.4f}')
print(f'Test accuracy     : , {score[1]:4.4f}')


# Définir la fonction plot_history
def plot_history(history):
    print("Plotting the training history")
    plt.figure(figsize=(6, 4))

    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Courbe de précision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.show()

# Appeler la fonction plot_history
#plot_history(history)



print("predicting the model")

y_sigmoid = model.predict(x_test)
y_pred = np.argmax(y_sigmoid, axis=-1)


# Définir la fonction plot_multiple_images_pred
def plot_multiple_images_pred(images, indices, columns=12, x_size=2, y_size=2, y_pred=None, saveas=None):
    rows = (len(indices) + columns - 1) // columns
    plt.figure(figsize=(columns * x_size, rows * y_size))
    for i, idx in enumerate(indices):
        plt.subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[idx].reshape(28, 28), cmap=plt.cm.binary)
        if y_pred is not None:
            plt.xlabel(f"{y_pred[idx]}", fontsize=12, color='red')
    if saveas:
        plt.savefig(saveas)
    plt.show()

# Afficher toutes les prédictions
#plot_multiple_images_pred(x_test, range(0, 200), columns=12, x_size=2, y_size=2, y_pred=y_pred, saveas='03-predictions')


print("Matrice de confusion")

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()
