from transforms_factory import get_transforms


# Build Your Datasets here based on the configuration
def get_datasets(config_data):
    x_transform, y_transform = get_transforms(config_data)
    data_dir = config_data['dataset']['path']
    train_fraction = config_data['dataset']['training_fraction']
    ds_train, ds_val = None, None

    raise Exception("Dataset Factory Not Implemented")
