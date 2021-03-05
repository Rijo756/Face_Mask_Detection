import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Using Inception_V3 as pretrained model
pre_trained_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

#all the pretrained layers are not trained again
for layer in pre_trained_model.layers:
    layer.trainable = False

#until mixed_7 we are choosing from inception 3
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

#adding last layer as our trainable fully conncected network with 3 class output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=x)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#loading input data
X = np.load('X.npy')
Y = np.load('Y.npy')

# Normalizing
X = X / 255.

#Defining test and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

#Train the model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model-{epoch:03d}.ckpt',
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True, 
    verbose=0)

history = model.fit(X_train, 
                    Y_train, 
                    epochs=20, 
                    callbacks=[checkpoint], 
                    validation_split=0.1)

#save the model
model.save('inceptionV3-model.h5')

#plot the accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(acc, label='Training')
plt.plot(val_acc, label='Validation')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(122)
plt.plot(loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#behaviour on test data
Y_pred = np.argmax(model.predict(X_test), axis=1)
Y_test = np.argmax(Y_test, axis=1)
sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='g', cmap=plt.cm.Blues)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()



