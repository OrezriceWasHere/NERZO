from tqdm import tqdm

import clearml_helper

N_PARTS = 20
BASE_TASK_NAME = "CascadeNER − NERtrieve Extraction"
PROJECT = "nertrieve_pipeline"
QUEUE = "a100_gpu"

if __name__ == "__main__":
    for i in tqdm([0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]):
        task = clearml_helper.get_task_by_description(
            description=BASE_TASK_NAME,
            new_project=PROJECT,
        )
        task.name = f"{BASE_TASK_NAME} part {i+1}/{N_PARTS}"
        conf = {"split_count": N_PARTS, "split_index": i, "batch_size": 100}
        task.connect(conf, name="split")
        task.enqueue(task, queue_name=QUEUE)
