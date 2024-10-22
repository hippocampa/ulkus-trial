from extractor import FeatureExtractor
import tensorflow_datasets as tfds
from keras import Sequential
from keras import layers, applications
from constants import IMG_SIZE
import tensorflow as tf


def augment_image(data, augmentor):
    image = data['image']
    label = data['label']
    return augmentor(image), label

def main():
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

    f_extractor = FeatureExtractor()
    f_extractor.register(applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet', pooling="avg"))
    f_extractor.register(applications.ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet', pooling="avg"))
    features, labels =f_extractor.extract(trainset)
    print(features)




if __name__ == "__main__":
    main()