import tensorflow
import numpy as np

tensorflow.experimental.numpy.experimental_enable_numpy_behavior(True)
class FeatureExtractor(object):
    """
    Feature extractor class
    """
    def __init__(self):
        self.models = []
        self.models_name = []
        self.features = []

    def register(self, model, name=None):
        model.trainable=False
        self.models.append(model)
        self.models_name.append(name if name else f"model_{len(self.models)}")

    # def extract(self, dataset):
    #     all_features = []
    #     all_labels = []

    #     @tensorflow.function
    #     def extract_batch(images):# -> dict[Any, Any]:
    #         features_dict = {}
    #         for model, name in zip(self.models, self.models_name):
    #             features_dict[name] = model(images)
    #         return features_dict
        
    #     for image, labels in dataset:
    #         batch_features_dict = extract_batch(image)
    #         batch_features = []
    #         for name in self.models_name:
    #             features = batch_features_dict[name].numpy() # type: ignore
    #             if len(features.shape) > 2:
    #                 features = features.reshape(features.shape[0], -1)
    #             batch_features.append(features)
    #         batch_features = np.concatenate(batch_features, axis=1)
    #         all_features.append(batch_features)
    #         all_labels.append(labels.numpy())
    #     features = np.concatenate(all_features, axis=0)
    #     labels = np.concatenate(all_labels, axis=0)

    #     return features, labels

    def extract(self, dataset):
        all_features = []
        all_labels = []

        @tensorflow.function
        def extract_batch(images):
            features_dict = {}
            for model, name in zip(self.models, self.models_name):
                features_dict[name] = model(images)
            return features_dict
        
        total_batches = tensorflow.data.experimental.cardinality(dataset).numpy()

        try:
            for batch_idx, (images, labels) in enumerate(dataset):
                print(f"Extracting batch {batch_idx+1}/{total_batches}", end='\r')
                batch_feature_dict = extract_batch(images)
                batch_features = []
                for name in self.models_name:
                    features = batch_feature_dict[name] # type: ignore
                    if len(features.shape) > 2:
                        features = features.reshape(features.shape[0], -1)
                    batch_features.append(features)
                batch_features = np.concatenate(batch_features, axis=1)
                all_features.append(batch_features)
                all_labels.append(labels.numpy())
        except tensorflow.errors.OutOfRangeError:
            print("\nFinished processing all batches")
        
        if not all_features:
            raise ValueError("No features extracted. Dataset might be empty")
        
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        print(f"\nExtracted features shape: {features.shape}")
        print(f"Extracted labels shape: {labels.shape}")

        return features, labels


