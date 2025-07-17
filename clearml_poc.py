import uuid

from clearml import Task, OutputModel, Model
import numpy as np
import pandas as pd
from runtime_args import RuntimeArgs


def clearml_allowed(func):
    def wrapper(*args, **kwargs):
        if RuntimeArgs.allow_clearml:
            return func(*args, **kwargs)

    return wrapper


@clearml_allowed
def clearml_init(project_name=None, task_name=None, requirements=None, queue_name=None):
    global execution_task, output_model
    Task.add_requirements('bitsandbytes', '>=0.43.2')
    Task.add_requirements('transformers', '==4.46.2')
    Task.add_requirements('torch', '==2.4.0')
    requirements = requirements or []
    for requirement in requirements:
        Task.add_requirements(requirement, '')
    # Task.add_requirements("requirements.txt", '')

    execution_task = Task.init(project_name=project_name or "NER - Zero Shot Chat GPT",
                               task_name=task_name or "hidden layers - match an entity to another sentence to detect same entity",
                               task_type=Task.TaskTypes.optimizer,
                               reuse_last_task_id=False)

    if execution_task.running_locally() and not task_name:
        name = input("enter description for task:\n")
        execution_task.set_name(name)


    queue_name = queue_name or RuntimeArgs.compute_queue

    if RuntimeArgs.running_remote:
        execution_task.execute_remotely(queue_name=queue_name,
                                        exit_process=True)


@clearml_allowed
def add_requirement(requirement, version=''):
    execution_task.add_requirements(requirement, version)


@clearml_allowed
def clearml_connect_hyperparams(hyperparams, name="general"):
    if hyperparams:
        execution_task.connect(hyperparams, name=name)

@clearml_allowed
def get_project_name():
    global execution_task
    project_name = execution_task.project

@clearml_allowed
def download_model(model_id):
    model = Model(model_id)
    return model.get_local_copy(raise_on_error=True)

@clearml_allowed
def get_clearml_task_id():
    return execution_task.id

@clearml_allowed
def clearml_display_image(image, iteration, series, description):
    execution_task.get_logger().report_image(description, image=image, iteration=iteration, series=series)


@clearml_allowed
def add_point_to_graph(title, series, x, y):
    execution_task.get_logger().report_scalar(title, series, value=y, iteration=x)

@clearml_allowed
def change_name(name):
    execution_task.set_name(name)


@clearml_allowed
def add_scatter(title, series, iteration, values):
    numpy_values = np.array(values)
    if numpy_values.ndim == 1:
        values = np.column_stack((np.arange(len(numpy_values)), numpy_values))
    execution_task.get_logger().report_scatter2d(title, series, scatter=values, iteration=iteration,
                                                 mode='lines+markers')


@clearml_allowed
def add_matplotlib(figure, iteration, series):
    execution_task.get_logger().report_matplotlib(figure, iteration=iteration, series=series)


@clearml_allowed
def add_confusion_matrix(matrix, title, series, iteration):
    execution_task.get_logger().report_confusion_matrix(title, series=series, matrix=matrix, iteration=iteration)


@clearml_allowed
def add_text(text):
    execution_task.get_logger().report_text(text)


@clearml_allowed
def add_table(title, series, iteration, table: pd.DataFrame):
    table.index.name = "id"
    execution_task.get_logger().report_table(title, series, iteration, table)

@clearml_allowed
def generate_tracked_model(**kwargs) -> OutputModel:
    kwargs = {**kwargs, "task":execution_task}
    model = OutputModel(**kwargs)
    model.connect(task=execution_task)
    return model




@clearml_allowed
def register_artifact(artifact, name):
    execution_task.register_artifact(artifact=artifact, name=name)

@clearml_allowed
def upload_model_to_clearml(model: OutputModel, model_path):
    if RuntimeArgs.upload_model:
        # if model.id:
        target_filename = str(uuid.uuid4())+ ".pt"
        output_uri = execution_task.storage_uri or execution_task._get_default_report_storage_uri()

        model.update_weights(model_path, upload_uri=output_uri)
        OutputModel.wait_for_uploads()
        # else:
        #     execution_task.update_output_model(model_path=model_path)

        print(f'uploading models from {model_path}')

@clearml_allowed
def add_tags(tags):
    execution_task.add_tags(tags)

