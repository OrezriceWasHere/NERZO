from clearml import Task, StorageManager
from clearml_pipelines.fewnerd_pipeline.fewnerd_dataset import  datasets

# create an dataset experiment
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 1 dataset artifact", )

# only create the task, we will actually execute it later
task.execute_remotely()

for dataset in datasets:
    url_dev = dataset["url"]
    name = dataset["name"]
    env = dataset["env"]

    print(f"Downloading dataset {name} from {url_dev}")

    # simulate local dataset, download one, so we have something local
    local_fewnerd = StorageManager.get_local_copy(
        remote_url=url_dev, name=name)
    print(f"Downloaded dataset {name} from {url_dev}. saved to {local_fewnerd}")
    # task.upload_artifact(f'raw-dataset-{env}', artifact_object=local_fewnerd)


# add and upload local file containing our toy dataset

print('uploading artifacts in the background')

# we are done
print('Done')