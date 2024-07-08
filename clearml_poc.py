from clearml import Task
from os import environ as env
import numpy as np

ALLOW_CLEARML = True if env.get("ALLOW_CLEARML") == "yes" else False
RUNNING_REMOTE = True if env.get("RUNNING_REMOTE") == "yes" else False


def clearml_init():
    global execution_task
    if ALLOW_CLEARML:

        Task.add_requirements("requirements.txt")
        execution_task = Task.init(project_name="NER - Hidden layers",
                                   task_name="hidden layers - same type vs different type",
                                   task_type=Task.TaskTypes.testing,
                                   reuse_last_task_id=False,)

        if execution_task.running_locally() and RUNNING_REMOTE:
            name = input("enter description for task:\n")
            execution_task.set_name(name)

        if RUNNING_REMOTE:
            execution_task.execute_remotely(queue_name="gpu", exit_process=True)


def clearml_display_image(image, iteration, series, description):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_image(description,

                                                 image=image,
                                                 iteration=iteration,
                                                 series=series)


def add_point_to_graph(title, series, x, y):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_scalar(title, series, value=y, iteration=x)


def add_scatter(title, series, iteration, values):
    if ALLOW_CLEARML:
        numpy_values = np.array(values)
        if numpy_values.ndim == 1:
            values = np.column_stack((np.arange(len(numpy_values)), numpy_values))
        execution_task.get_logger().report_scatter2d(title, series, scatter=values, iteration=iteration,
                                                     mode='lines+markers')


def add_matplotlib(figure, iteration, series):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_matplotlib(figure, iteration=iteration, series=series)


def add_confusion_matrix(matrix, title, series, iteration):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_confusion_matrix(title, series=series, matrix=matrix, iteration=iteration)


def add_text(text):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_text(text)
