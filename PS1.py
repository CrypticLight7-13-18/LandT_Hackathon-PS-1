import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from keras.layers import Dropout
from tensorflow.keras.models import Sequential

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

valid_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

valid_set = valid_datagen.flow_from_directory('valid',
                                              target_size = (64, 64),
                                              batch_size = 32,
                                              class_mode = 'binary')
base_model = keras.applications.ResNet50(weights='imagenet',
                                      include_top=False,
                                      input_shape=(64, 64, 3))

model = keras.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.add(Dropout(0.2))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(training_set, epochs = 65, validation_data = valid_set)
test_loss, test_acc = model.evaluate(test_set)
print('Test accuracy:', test_acc)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Get predictions for test set
predictions = model.predict(test_set)
predictions = [1 if p > 0.5 else 0 for p in predictions]

# Create confusion matrix
cm = confusion_matrix(test_set.classes, predictions)

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(test_set.classes, predictions, average='binary')

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
