import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np


# SET ALL CONSTRAINTS
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=50

# IMPORT DATA INTO TENSORFLOW DATASET OBJECT
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Plant_village",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Defining class names 
class_names = dataset.class_names
class_names

##

##
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())

# VISUALIZE SOME OF THE IMAGES FROM OUR DATASET

# To store this train, test, split we will make it into function
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
  
# shuffle_ -- DS should be randomly shuffled before partitioning it into train, validation n test sets
# assert -- This assertion is checking whether the sum of the train_split, val_split, and test_split values equals 1

    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# Split the dataset into train, validation, and test sets
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# CACHE, SHUFFLE AND PREFETCH THE DATA
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# BUILDING THE MODEL 
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])


# Data Augmentation
# Data Augmentation is needed when we have less data, this boosts the accuracy of our model by augmenting the data.

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

# APPLYING DATA AUGMENTATION TO TRAIN DATASET
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# MODEL ARCHITECTURE
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])
model.build(input_shape=input_shape)

# COMPILING THE MODEL 
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=50,
)


# PREDICTING IMAGE FORM DATA
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])



# PREDICT MODEL IN FORM OF CONFIDENCE AND PREDICTED CLASS
#def predict(model, img):
    #img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    #img_array = tf.expand_dims(img_array, 0)

    #predictions = model.predict(img_array)

    #predicted_class = class_names[np.argmax(predictions[0])]
    #confidence = round(100 * (np.max(predictions[0])), 2)
    #return predicted_class, confidence


# SAVE MODEL IN FILE FORMAT I.E. JSON/H5/
# Assuming 'model' is your Keras model

# Convert the model to JSON format
model_json = model.to_json()

# Save the JSON to a file
model_json_path = "model.json"
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

print(f"Model saved as JSON: {model_json_path}")


# %%%%%%%%%% H5

model.save("potatoes.h5")
print("Saved model to disk")


#model.fit_generator(train_ds,
 #                        steps_per_epoch = ,
 #                        epochs = 50,
 #                        validation_data=val_ds
#
 #                        )

#classifier_json=model.to_json()
#with open("model.json", "w") as json_file:
  #  json_file.write(classifier_json)

    # serialize weights to HDF5
   # model.save_weights("my_model_weights.h5")
   # model.save("model.h5")
   # print("Saved model to disk")