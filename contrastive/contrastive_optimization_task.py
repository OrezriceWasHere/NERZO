import logging

from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
ParameterSet,
LogUniformParameterRange,
    UniformIntegerParameterRange)
from clearml.automation import RandomSearch # noqa

# trying to load Bayesian optimizer package
aSearchStrategy = RandomSearch

#
# def all_optimizer_hyperparam() -> list:
#     return

    # args.input_methods = trial.suggest_categorical('input_methods', ['start_end_pair', 'end', 'diff'])



    # args.is_hidden_layer = trial.suggest_categorical('is_hidden_layer', [True, False])
    # args.output_size = trial.suggest_int('output_size', 20, 500)
    # args.hidden_size = trial.suggest_int('hidden_size', 20, 500)

    # args.lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    # args.dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # args.activation = trial.suggest_categorical('activation', ['silu', 'leaky_relu', 'relu'])
    # args.noise = trial.suggest_categorical('noise', ['dropout', 'identity'])
    # args.loss_fn = trial.suggest_categorical('loss_fn', ['triplet_loss', 'dpr_loss'])


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='Hyper-Parameter Optimization',
                 task_name='Automatic Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

# experiment template to optimize in the hyper-parameter optimization
args = {
    'template_task_id': None,
    'run_as_service': True,
}
args = task.connect(args)

# Get the template task experiment that we want to optimize
if not args['template_task_id']:
    args['template_task_id'] = 'a5e34dd5a53a4541a2aaf09b07843e0a'

# Set default queue name for the Training tasks themselves.
# later can be overridden in the UI
execution_queue = 'dsicsgpu'

# Example use case:
an_optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id=args['template_task_id'],
    # here we define the hyper-parameters to optimize
    # Notice: The parameter name should exactly match what you see in the UI: <section_name>/<parameter>
    # For Example, here we see in the base experiment a section Named: "General"
    # under it a parameter named "batch_size", this becomes "General/batch_size"
    # If you have `argparse` for example, then arguments will appear under the "Args" section,
    # and you should instead pass "Args/batch_size"

    hyper_parameters=[
        UniformIntegerParameterRange('general/hidden_layer', min_value=50, max_value=250, step_size=25),
        UniformIntegerParameterRange('general/output_layer', min_value=50, max_value=250, step_size=25),
        DiscreteParameterRange('general/lr', values=[2e-06]),

        DiscreteParameterRange('general/activation', values=['silu', 'relu']),
        # DiscreteParameterRange('general/noise', values=['dropout', 'identity']),
        DiscreteParameterRange('general/loss_fn', values=['triplet_loss', 'contrastive_loss']),
        DiscreteParameterRange('general/llm_layer', values=['llama_3_17_v_proj', 'llama_3_3_13_k_proj']),
        DiscreteParameterRange('general/input_tokens', values=['start_end_pair', 'end']),
        DiscreteParameterRange('general/is_hidden_layer', values=[True, False]),
        DiscreteParameterRange('general/dropout', values=[0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        DiscreteParameterRange('general/hard_negative_ratio', values=[0.1, 0.15, 0.2, 0.25, 0.5, 1]),
        DiscreteParameterRange('general/triplet_loss_margin', values=[0.2, 0.5, 0.65, 0.8, 0.9, 1.0]),

    ],
    # this is the objective metric we want to maximize/minimize
    objective_metric_title='auc',
    objective_metric_series='eval',
    # now we decide if we want to maximize it or minimize it (accuracy we maximize)
    objective_metric_sign='max_global',
    # let us limit the number of concurrent experiments,
    # this in turn will make sure we do dont bombard the scheduler with experiments.
    # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
    max_number_of_concurrent_tasks=8,
    # this is the optimizer class (actually doing the optimization)
    # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
    # more are coming soon...
    optimizer_class=aSearchStrategy,
    # Select an execution queue to schedule the experiments for execution
    execution_queue=execution_queue,
    # If specified all Tasks created by the HPO process will be created under the `spawned_project` project
    spawn_project='Optimizer Tasks',  # 'HPO spawn project',
    # If specified only the top K performing Tasks will be kept, the others will be automatically archived
    save_top_k_tasks_only=None,  # 5,
    # Optional: Limit the execution time of a single experiment, in minutes.
    # (this is optional, and if using  OptimizerBOHB, it is ignored)
    time_limit_per_job=None,
    # Check the experiments every 12 seconds is way too often, we should probably set it to 5 min,
    # assuming a single experiment is usually hours...
    # pool_period_min=5,
    # set the maximum number of jobs to launch for the optimization, default (None) unlimited
    # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
    # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
    total_max_jobs=40,
    # set the minimum number of iterations for an experiment, before early stopping.
    # Does not apply for simple strategies such as RandomSearch or GridSearch
    min_iteration_per_job=10,
    # Set the maximum number of iterations for an experiment to execute
    # (This is optional, unless using OptimizerBOHB where this is a must)
    max_iteration_per_job=200
)

# if we are running as a service, just enqueue ourselves into the services queue and let it run the optimization
if args['run_as_service']:
    # if this code is executed by `clearml-agent` the function call does nothing.
    # if executed locally, the local process will be terminated, and a remote copy will be executed instead
    task.execute_remotely(queue_name='cpu', exit_process=True)

# report every 12 seconds, this is way too often, but we are testing here
an_optimizer.set_report_period(20)
# start the optimization process, callback function to be called every time an experiment is completed
# this function returns immediately
an_optimizer.start(job_complete_callback=job_complete_callback)
# You can also use the line below instead to run all the optimizer tasks locally, without using queues or agent
# an_optimizer.start_locally(job_complete_callback=job_complete_callback)
# set the time limit for the optimization process (2 hours)
# wait until process is done (notice we are controlling the optimization process in the background)
an_optimizer.wait()
# optimization is completed, print the top performing experiments id
top_exp = an_optimizer.get_top_experiments(top_k=5)
print([t.id for t in top_exp])
# make sure background optimization stopped
an_optimizer.stop()

print('We are done, good bye')