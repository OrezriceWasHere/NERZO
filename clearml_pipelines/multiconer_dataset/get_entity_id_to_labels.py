import clearml_helper


if __name__ == "__main__":
	task = clearml_helper.get_task_by_description(
		description="text id to all labels",
		new_project="multiconer_pipeline"
	)

	task.name = "text id to all labels"

	conf = {"layer_id": "7b24211634fd454e99d34b65286ab4d7",
	"slow_down_intentionally": False,
	"elasticsearch_index": "multiconer_validation,multiconer_test,multiconer_train"}

	task.connect(conf, name="conf")
	task.enqueue(task, queue_name="dsicsgpu")
