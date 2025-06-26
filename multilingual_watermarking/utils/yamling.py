import yaml


def read_yaml(file_path):
    """Read a YAML file and return its contents as a Python object."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_yaml(data, file_path):
    """Write a Python object to a YAML file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)