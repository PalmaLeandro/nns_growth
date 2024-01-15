import json 

def persist_experiment(base_path, experiment_specification): 
    file_name = base_path
    file_name += f'seed_{experiment_specification["seed"]}_'
    file_name += f'input_dimension_{experiment_specification["input_dimension"]}_'
    file_name += f'output_dimension_{experiment_specification["output_dimension"]}_'
    file_name += f'sample_size_{experiment_specification["sample_size"]}_'
    file_name += f'epochs_{experiment_specification["epochs"]}_'
    file_name += f'learning_rate_{experiment_specification["learning_rate"]}_'
    file_name += '.json'
    
    with open(file_name, "w", encoding="utf8") as fp:
        json.dump(experiment_specification, fp, indent=2)

def load_experiment(path):
    with open(path, 'r') as fp:
        experiment_specification = json.load(fp)
        
    return experiment_specification