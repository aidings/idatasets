import os
import yaml

data_root = os.path.join(os.path.dirname(__file__), 'data')


class VisualFormat:
    def __init__(self, conf_file=None):
        conf = conf_file or os.path.join(data_root, 'format.yaml')
        with open(conf, 'r') as f:
            self.conf_dict = yaml.load(f, yaml.FullLoader) 

        self.formats = {}
        for key in self.conf_dict['format'].keys():
            self.formats[key] = [ext.upper() for ext in self.conf_dict['format'][key]]
            self.formats[key].extend(self.conf_dict['format'][key])
            self.formats[key] = set(self.formats[key])

    def __getitem__(self, key):
        return self.formats[key]


gformat = VisualFormat()


from .utils import *