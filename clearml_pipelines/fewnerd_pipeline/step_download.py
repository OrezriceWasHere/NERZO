from clearml import Task, StorageManager

# create an dataset experiment
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 1 dataset artifact")

# only create the task, we will actually execute it later
task.execute_remotely()

datasets = [
    {
        "url": "https://huggingface.co/datasets/Rosenberg/fewnerd/resolve/main/dev-supervised.txt",
        "name": "dev-supervised.txt",
        "json": "dev-supervised.json",
        "env": "dev"
    },
    {
        "url": "https://huggingface.co/datasets/Rosenberg/fewnerd/resolve/main/train-supervised.txt",
        "name": "train-supervised.txt",
        "json": "train-supervised.json",
        "env": "train"
    },
    {
        "url": "https://huggingface.co/datasets/Rosenberg/fewnerd/resolve/main/test-supervised.txt",
        "name": "test-supervised.txt",
        "json": "test-supervised.json",
        "env": "test"
    }
]

for dataset in datasets:
    url_dev = dataset["url"]
    name = dataset["name"]
    env = dataset["env"]

    # simulate local dataset, download one, so we have something local
    local_fewnerd = StorageManager.get_local_copy(
        remote_url=url_dev, name=name)
    # task.upload_artifact(f'raw-dataset-{env}', artifact_object=local_fewnerd)


# add and upload local file containing our toy dataset

print('uploading artifacts in the background')

# we are done
print('Done')