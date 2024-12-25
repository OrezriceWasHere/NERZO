from clearml.automation import PipelineController
from os import environ as env

def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    packages="requirements.txt",
    name="Indexing FEWNERD Dataset into elasticsearch",
    project="fewnerd_pipeline", version="0.0.1", add_pipeline_tags=False
)
pipe.set_default_execution_queue('cpu')

pipe.add_step(
    name="stage_download",
    base_task_project="fewnerd_pipeline",
    base_task_name="Pipeline step 1 dataset artifact",
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
    cache_executed_step=True,
)

pipe.add_step(
    name="stage_process",
    parents=["stage_download"],
    base_task_project="fewnerd_pipeline",
    base_task_name="Pipeline step 2 jsonify dataset",
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
    cache_executed_step=True,
    execution_queue="a100_gpu",
)

pipe.add_step(
    name="stage_index_to_database",
    parents=["stage_process"],
    base_task_project="fewnerd_pipeline",
    base_task_name="Pipeline step 3 index to database",
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
)


# for debugging purposes use local jobs

SHOULD_DEPLOY = env.get("RUNNING_REMOTE", "no") == "yes"

if SHOULD_DEPLOY:
    pipe.start(queue='a100_gpu')
else:
    pipe.start_locally(run_pipeline_steps_locally=True)

# Starting the pipeline (in the background)

print("done")