from clearml.automation import PipelineController
from os import environ as env

from runtime_args import RuntimeArgs


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
    packages="./requirements.txt",
    name="Evaluating One shot retrieval",
    project="fewnerd_pipeline", version="0.0.1", add_pipeline_tags=True
)
pipe.add_parameter(name="mlp_id", description="mlp head id from clearml task or llm id", default="xxx")

pipe.set_default_execution_queue('dsicsgpu')

pipe.add_step(
    name="Step Index to Elasticsearch DB",
    base_task_project="fewnerd_pipeline",
    base_task_name="Pipeline step 4 calculate ne embedding",
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
    parameter_override={
        "layer_name/layer_id":pipe.get_parameters()["mlp_id"]
    },
    cache_executed_step=True,
)

pipe.add_step(
    name="Step Evaluate Recall",
    parents=["Step Index to Elasticsearch DB"],
    base_task_project="NER - Zero Shot Chat GPT",
    base_task_name="calculate recall",
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
    cache_executed_step=True,
    parameter_override={
        "layer_name/layer_id" :pipe.get_parameters()["mlp_id"]
    }
)



# for debugging purposes use local jobs

SHOULD_DEPLOY = RuntimeArgs.running_remote

if SHOULD_DEPLOY:
    pipe.start(queue='dsicsgpu')
else:
    pipe.start_locally(run_pipeline_steps_locally=True)

# Starting the pipeline (in the background)

print("done")