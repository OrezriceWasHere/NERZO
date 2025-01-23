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
def clearml_init(task_name=None):
    global execution_task, output_model
    Task.add_requirements('bitsandbytes', '>=0.43.2')
    Task.add_requirements('transformers', '>=4.45.0')
    Task.add_requirements('torch', '==2.4.0')
    Task.add_requirements('aiohttp')


    execution_task = Task.init(project_name="NER - Zero Shot Chat GPT",
                               task_name=task_name or "hidden layers - match an entity to another sentence to detect same entity",
                               task_type=Task.TaskTypes.optimizer,

                               reuse_last_task_id=False)
    if execution_task.running_locally() and not task_name:
        name = input("enter description for task:\n")
        execution_task.set_name(name)




    if RuntimeArgs.running_remote:
        execution_task.execute_remotely(queue_name=RuntimeArgs.compute_queue,

                                        exit_process=True)

@clearml_allowed
def clearml_connect_hyperparams(hyperparams, name="general"):
    if hyperparams:
        execution_task.connect(hyperparams, name=name)

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
    return OutputModel(task=execution_task, **kwargs)


@clearml_allowed
def register_artifact(artifact, name):
    execution_task.register_artifact(artifact=artifact, name=name)

@clearml_allowed
def upload_model_to_clearml(model: OutputModel, model_path):
    if RuntimeArgs.upload_model:
        model.update_weights(weights_filename=model_path)
        execution_task.update_output_model(model_path=model_path)
        OutputModel.wait_for_uploads()
        print(f'uploading models from {model_path}')

