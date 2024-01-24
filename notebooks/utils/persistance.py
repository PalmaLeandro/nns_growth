import os, json

def check_path(path):
    if not os.path.exists(path): os.makedirs(path)

def parameters_from_file_path(file_path):
    file_name = file_path.split(os.path.sep)[-1]
    file_name_without_extension = '.'.join(file_name.split('.')[:-1])

    # Split name, extract the values and names, and build a dictionary by using them as keys and values
    parameters_names_and_values = list(filter(lambda x: x, file_name_without_extension.split('_')))

    values_indices = [
        index for index, string in enumerate(parameters_names_and_values) 
        if string.replace('.', '', 1).isdigit()
    ]
    parameters_values = list(filter(lambda string: string.replace('.', '', 1).isdigit(), parameters_names_and_values))

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
    return {key: value for key, value in parameters.items() if key in parameter_names}

def file_name_from_parameters(parameters):
    return '_'.join([f'{key}_{value}'for key, value in parameters.items()])

def file_path_from_parameters(parameters, parameter_names, prefix='', suffix=''):
    file_name = file_name_from_parameters(pick_parameters(parameters, parameter_names))
    return f'{prefix}{file_name}{suffix}'

def save_model(model, model_parameters, models_path, name_parameters):
    import torch
    
    check_path(models_path)
    model_file_path = file_path_from_parameters(model_parameters, name_parameters, models_path, '.pt')
    torch.save(model.state_dict(), model_file_path)

def load_model(model_file_path, model_class=None, device=None):
    import torch

    model_parameters = parameters_from_file_path(model_file_path)
    if model_class is not None:
        model = model_class(**model_parameters)  # Recommended way to load a model according to PyTorch tutorial.
        model.load_state_dict(torch.load(model_file_path))

    else:
        model = torch.load(model_file_path)

    return model.to(device) if device is not None else model

def save_experiment(experiment, experimets_path, name_parameters): 
    check_path(experimets_path)
    experiment_file_path = file_path_from_parameters(experiment, name_parameters, experimets_path, '.pt')
    with open(experiment_file_path, 'w', encoding='utf8') as fp:
        json.dump(experiment, fp, indent=2)

def load_experiment(experiment_file_path):
    with open(experiment_file_path, 'r') as fp:
        experiment = json.load(fp)

    return experiment
