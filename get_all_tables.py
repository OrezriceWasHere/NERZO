import json

from clearml import Task
import pandas as pd
import io

def get_task_by_name(task_name, project_name=None):
    tasks = Task.query_tasks(project_name=project_name, task_name=task_name)
    if tasks:
        return Task.get_task(task_id=tasks[0])
    return None

def fetch_table_from_plot_variant(task, metric, variant):
    """
    Fetch a table from ClearML Plots section using its metric and variant.

    Args:
        task (Task): ClearML Task object.
        metric (str): metric of the plot group.
        variant (str): variant name of the table inside the plot.

    Returns:
        pd.DataFrame or None
    """
    plot_data = task.get_reported_plots()
    for tbl in plot_data:
        if tbl['metric'] == metric and tbl['variant'] == variant:
            # Table data is usually in CSV format
            try:
                data = json.loads(tbl['plot_str'])
                table_data = data['data'][0]

                # Flatten the headers
                raw_headers = table_data['header']['values']
                headers = [h[0] if isinstance(h, list) else h for h in raw_headers]

                # Transpose the cell values so rows align with headers
                rows = list(zip(*table_data['cells']['values']))

                # Convert to a DataFrame
                df = pd.DataFrame(rows, columns=headers)
                return df
            except Exception as e:
                print(f"Failed to parse table '{variant}' in '{metric}': {e}")
                return None
    print(f"Table with metric '{metric}' and variant '{variant}' not found in task '{task.name}'")
    return None

# Example usage
# experiment_names = ['eval nertrieve bm25',
#                     "eval nertrieve llama end",
#                     "eval nertrieve intfloat/e5-mistral-7b-instruct",
#                     "eval nertrieve f77",
#                     "eval nertrieve eos",
#                     "eval nertrieve bm25"]
# experiment_names = [
#     "calculate recall types only fewnerd",
#     "calculate recall fewnerd llama eos",
#     "calculate recall fewnerd llama entire model",
#     "calculate recall fewnerd end",
#     "calculate recall fewnerd Nvidia embedder",
#     "calculate recall e5 mistral fewnerd",
#     "calculate recall BM25 fewnerd"
#     ]
experiment_names = [
    "eval multiconer with sentence embedder nv-embed-v2",
    "eval multiconer mlp f77",
    "eval multiconer eos",
    "eval multiconer end",
    "eval multiconer bm25",
    "eval multiconer E5 Mistral",
    ]
project_name = 'publish'  # or specify your ClearML project
metric = "recall per fine type"
variant = "zero shot"

indices = []
serieses = []
for exp_name in experiment_names:
    task = get_task_by_name(exp_name, project_name)
    if task:
        df = fetch_table_from_plot_variant(task, metric, variant)
        df.index = df.id
        df_sorted = df.sort_index()
        precision_series = df_sorted['recall@size']
        serieses.append(precision_series)
        # Sort each dataframe by its index
    else:
        print(f"No task found with name '{exp_name}'")

result = pd.concat(serieses, axis=1)
result.index = df_sorted.id
result.columns = experiment_names
result.to_csv("multiconer.csv", index=True)
pass