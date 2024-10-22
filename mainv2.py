import tensorflow_datasets as tfds
from keras import Sequential
from keras import layers, applications
from constants import IMG_SIZE
import tensorflow as tf
import matplotlib.pyplot as plt


def augment_image(data, augmentor):
    image = data['image']
    label = data['label']
    return augmentor(image), label

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    builder = tfds.ImageFolder(r"data\klasifikasiDFU\klasifikasiDFU\Dataset Image")
    trainset = builder.as_dataset(split='Train Img', shuffle_files=True)
    testset = builder.as_dataset(split='Test Img', shuffle_files=False)

    # Augmentation
    augmentor = Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    # Apply augmentation to the dataset
    trainset = trainset.map(lambda data: augment_image(data, augmentor)) #type: ignore
    testset = testset.map(lambda data: augment_image(data, augmentor)) #type: ignore

    # Prepare the dataset for training
    BATCH_SIZE = 32
    trainset = trainset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    testset = testset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Training model
    mobilenet_v2 = applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    mobilenet_v2.trainable = False
    
    # Classification heads
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense(2, activation='sigmoid')

    model = Sequential([
        layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        mobilenet_v2,
        global_average_layer,
        prediction_layer,
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainset, validation_data=testset, epochs=5)

    # print(model.summary())

if __name__ == "__main__":
    main()