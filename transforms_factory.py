import torchvision as tv


# Build Your Transforms here based on the configuration
def get_transforms(config_data):
    x_out = []
    y_out = []

    for transform in config_data['dataset']['transforms']:
        if transform == 'Normalize':
            x_out.append(tv.transforms.Normalize())

    x_out.append(tv.transforms.ToTensor())
    y_out.append(tv.transforms.ToTensor())

    x_transform, y_transform = tv.transforms.Compose(x_out), tv.transforms.Compose(y_out)

    raise Exception("Transform Factory Not Implemented!!")
