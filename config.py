import yaml

def read_config():
    with open('config.yml', 'r') as cfg_yml:
        config = yaml.safe_load(cfg_yml)
    return config
