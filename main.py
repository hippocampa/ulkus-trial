import tensorflow_datasets as tfds
from keras import Sequential
from keras import layers, applications
from constants import IMG_SIZE
import tensorflow as tf
import matplotlib.pyplot as plt

def augment_image(data):
    image = data['image']
    label = data['label']
    augmentor = Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])
    return augmentor(image), label

def main():
    builder = tfds.ImageFolder(r"data\klasifikasiDFU\klasifikasiDFU\Dataset Image")
    trainset = builder.as_dataset(split='Train Img', shuffle_files=True)
    testset = builder.as_dataset(split='Test Img', shuffle_files=False)

    # Apply augmentation to the dataset
    trainset = trainset.map(augment_image)
    testset = testset.map(augment_image)

    # Visualize the augmented image
    sample_img, _ = next(iter(trainset))  # Get a sample image
    plt.imshow(tf.squeeze(sample_img))  # Squeeze to remove batch dimension
    plt.axis('off')  # Hide axes for better visualization
    plt.show()  # Display the plot

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

    print(model.summary())

if __name__ == "__main__":
    main()