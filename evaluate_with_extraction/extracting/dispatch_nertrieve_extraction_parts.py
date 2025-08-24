from tqdm import tqdm

import clearml_helper

N_PARTS = 20
BASE_TASK_NAME = "CascadeNER − NERtrieve Extraction"
PROJECT = "nertrieve_pipeline"
QUEUE = "slurm_a100"

if __name__ == "__main__":
    for i in tqdm(range(N_PARTS), desc="Creating tasks"):
        task = clearml_helper.get_task_by_description(
            description=BASE_TASK_NAME,
            new_project=PROJECT,
        )
        task.name = f"{BASE_TASK_NAME} part {i+1}/{N_PARTS}"
        conf = {"split_count": N_PARTS, "split_index": i, "batch_size": 50}
        task.connect(conf, name="split")
        task.enqueue(task, queue_name=QUEUE)
