from collections import defaultdict


def insert_node(tree, path, max_depth=2):
    parts = path.split('.')
    current_level = tree
    for i, part in enumerate(parts):
        if i >= max_depth:  # If the depth limit is reached, store the remaining parts as a value
            remaining_path = '.'.join(parts[i:])
            current_level[remaining_path] = {}
            break
        if part not in current_level:
            current_level[part] = {}
        current_level = current_level[part]

def build_tree(data, max_depth=2):
    tree = {}
    for item in data:
        insert_node(tree, item, max_depth)
    return tree







def get_all_nodes_with_parents(tree, parent=None):
    nodes_with_parents = {}
    for key, value in tree.items():
        current_path = f"{parent}.{key}" if parent else key
        nodes_with_parents[current_path] = parent
        if isinstance(value, dict):
            nodes_with_parents.update(get_all_nodes_with_parents(value, current_path))
    return nodes_with_parents


def map_item_to_group(list_of_values, max_depth=3):
    tree = build_tree(list_of_values, max_depth)
    unique_parents = set(get_all_nodes_with_parents(tree).values())
    parent_to_group_id = {parent: i for i, parent in enumerate(unique_parents)}
    item_to_group_id = {item: parent_to_group_id[parent] for item, parent in get_all_nodes_with_parents(tree).items()}
    return item_to_group_id

def group_layers(list_of_values, max_depth=3):
    item_to_group = map_item_to_group(list_of_values, max_depth)
    grouped_layers = defaultdict(list)
    for item, group_id in item_to_group.items():
        grouped_layers[group_id].append(item)
    return dict(grouped_layers)


if __name__ == "__main__":
    data = [
        "root",
        "root.child1",
        "root.child1.grandchild1",
        "root.child1.grandchild1.greatgrandchild1",
        "root.child2",
        "root.child2.grandchild2",
        "root.child2.grandchild3",
        "root.child2.grandchild2.extra"
    ]

    print(dict(group_layers(data)).values())