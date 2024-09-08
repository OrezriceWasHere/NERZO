from clearml import Task
from os import environ as env
import numpy as np
import pandas as pd

ALLOW_CLEARML = True if env.get("ALLOW_CLEARML") == "yes" else False
RUNNING_REMOTE = True if env.get("RUNNING_REMOTE") == "yes" else False


def clearml_allowed(func):
    def wrapper(*args, **kwargs):
        if ALLOW_CLEARML:
            return func(*args, **kwargs)

    return wrapper


@clearml_allowed
def clearml_init():
    global execution_task
    Task.add_requirements("requirements.txt")
    execution_task = Task.init(project_name="NER - Zero Shot Chat GPT",
                               task_name="hidden layers - match an entity to another sentence to detect same entity",
                               task_type=Task.TaskTypes.testing,
                               reuse_last_task_id=False)

    if execution_task.running_locally():
        name = input("enter description for task:\n")
        execution_task.set_name(name)

    if RUNNING_REMOTE:
        execution_task.execute_remotely(queue_name="gpu", exit_process=True)


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
