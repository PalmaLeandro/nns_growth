import os, json

def check_path(path):
    if not os.path.exists(path): os.makedirs(path)

def is_numeric(string):
    return string.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit()

def parameters_from_file_path(file_path):
    file_name = file_path.split(os.path.sep)[-1]
    file_name_without_extension = '.'.join(file_name.split('.')[:-1])

    # Split name, extract the values and names, and build a dictionary by using them as keys and values
    parameters_names_and_values = list(filter(lambda x: x, file_name_without_extension.split('_')))

    values_indices = [index for index, string in enumerate(parameters_names_and_values) if is_numeric(string)]
    parameters_values = list(filter(is_numeric, parameters_names_and_values))

    names_indices_from = [0] + [value_index + 1 for value_index in values_indices]
    names_indices_to = values_indices + [len(parameters_names_and_values)]

    parameter_names = [
        '_'.join(parameters_names_and_values[index_from: index_to]) 
        for index_from, index_to in zip(names_indices_from, names_indices_to)
    ]

    return {
        parameter_name: float(parameter_value) if '.' in parameter_value else int(parameter_value)
        for parameter_name, parameter_value in zip(parameter_names, parameters_values)
    }

def pick_parameters(parameters, parameter_names):
    return {parameter: parameters[parameter] for parameter in parameter_names}

def file_name_from_parameters(parameters):
    return '_'.join([f'{key}_{value}'for key, value in parameters.items()])

def file_path_from_parameters(parameters, parameter_names=None, prefix='', suffix=''):
    parameters = parameters if parameter_names is None else pick_parameters(parameters, parameter_names)
    file_name = file_name_from_parameters(parameters)
    return f'{prefix}{file_name}{suffix}'

def save_model(model, models_path):
    import torch
    
    check_path(models_path)
    model_file_path = file_path_from_parameters(model.persitance_parameters, 
                                                prefix=models_path, 
                                                suffix=model.FILE_EXTENSION)
    torch.save(model.state_dict(), model_file_path)

def load_model(model_file_path, model_class=None, device=None, extra_model_parameters={}):
    import torch

    model_parameters = parameters_from_file_path(model_file_path)
    model_parameters.update(extra_model_parameters)
    if model_class is not None:
        model = model_class(**model_parameters)  # Recommended way to load a model according to PyTorch tutorial.
        model.load_state_dict(torch.load(model_file_path))

    else:
        model = torch.load(model_file_path)

    return model.to(device) if device is not None else model

def save_experiment(experiment, experimets_path, name_parameters): 
    check_path(experimets_path)
    experiment_file_path = file_path_from_parameters(experiment, name_parameters, experimets_path, '.json')
    with open(experiment_file_path, 'w', encoding='utf8') as fp:
        json.dump(experiment, fp, indent=2)

def load_experiment(experiment_file_path):
    with open(experiment_file_path, 'r') as fp:
        experiment = json.load(fp)

    return experiment


class PersistableModel(object):

    MODEL_NAME_PARAMETERS = []
    FILE_EXTENSION = '.pt'

    def store_parameters(self, **kwargs):
        self.persitance_parameters = {parameter: kwargs[parameter] for parameter in self.MODEL_NAME_PARAMETERS}
    
    def save(self, path):
        save_model(self, path)

    @classmethod
    def load(cls, path, parameters=None):
        if parameters is not None: # path specifies a folder containing the model.
            path = file_path_from_parameters(parameters, cls.MODEL_NAME_PARAMETERS, prefix=path, suffix=cls.FILE_EXTENSION)

        return load_model(path, cls)
    