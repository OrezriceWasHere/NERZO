from clearml import Task

ALLOW_CLEARML = True
RUNNING_REMOTE = True

if ALLOW_CLEARML:
    execution_task = Task.init(project_name="NER - Zero Shot Chat GPT",
                               task_name="Running on server",
                               task_type=Task.TaskTypes.optimizer,
                               reuse_last_task_id=False)
    if RUNNING_REMOTE:
        execution_task.execute_remotely(queue_name="cpu", exit_process=True)
