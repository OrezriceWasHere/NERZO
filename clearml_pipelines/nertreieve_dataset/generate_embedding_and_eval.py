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
    name="Evaluating an mlp",
    project="NER - Zero Shot Chat GPT", version="0.0.1", add_pipeline_tags=False
)

pipe.add_parameter(
    name='mlp_head',
    description='MLP head id',
    default='6b11b974e63543eb942741562046c063'
)

pipe.add_step(
    name="stage_generate_ne_embedding",
	base_task_id='1dfe63a8766b49a2bcdc575fcdd76a54',
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
	parameter_override={
		'conf/layer_id': '${pipeline.mlp_head}',
	},
	execution_queue='dsicsgpu'
)

pipe.add_step(
    name="stage_eval",
    parents=["stage_generate_ne_embedding"],
	base_task_id='2ee7ddfc94b849e3b299635afb540678',
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
	parameter_override={
		'eval nertrieve/layer_id': '${pipeline.mlp_head}',
	},
	execution_queue='dsicsgpu'
)



# for debugging purposes use local jobs

SHOULD_DEPLOY = env.get("RUNNING_REMOTE", "no") == "yes"

if SHOULD_DEPLOY:
    pipe.start(queue='dsicsgpu')
else:
    pipe.start_locally(run_pipeline_steps_locally=True)

# Starting the pipeline (in the background)

print("done")