import os
import yaml
from munch import DefaultMunch

"""
Load settings of MixerCaps from MixerCaps.yaml
"""

ROOT_PATH = '/home/liuxiaolong/IQA/MixerCaps'
current_path = 'config'
config_file_name = 'MixerCaps.yaml'

config_file_path = os.path.join(ROOT_PATH, current_path, config_file_name)
yaml_dict = yaml.load(open(config_file_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
cfg = DefaultMunch.fromDict(yaml_dict)

