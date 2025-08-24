from clearml.backend_interface import Task

import clearml_helper
import clearml_poc


if __name__ == "__main__":
	task = clearml_helper.get_task_by_description(
		description="Pipeline step 4 calculate ne embedding nertrieve",
		new_project="multiconer_pipeline"
	)

	task.name = "multiconer calcualte ne retrieve"

	conf = {
		"index":"multiconer_validation,multiconer_test,multiconer_train",
		"entity_type_key": "entity_type"
	}

	task.connect(conf, name="conf")
	task.enqueue(task, queue_name="dsicsgpu")
