import fiftyone as fo

name = "my-dataset"
dataset_dir = "C:/Users/xyche/Downloads/dataset"

# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    name=name,
)

# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())

session = fo.launch_app(dataset)