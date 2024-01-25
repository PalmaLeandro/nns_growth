import os, sys, time, numpy, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test
from settings.two_layer_teacher_network import fresh_dataset_dataloader

EXPERIMENT_NAME_PARAMETERS = ['seed', 'input_dimension', 'output_dimension', 'sample_size', 'batch_size', 
                              'epochs', 'learning_rate']
MODEL_NAME_PARAMETERS = EXPERIMENT_NAME_PARAMETERS + ['overparametrization', 'run']                                      

def assertion(input_dimension, output_dimension, sample_size, batch_size, device):
    dataloader, true_output_weights = fresh_dataset_dataloader(input_dimension, sample_size, batch_size)
    model = OverparametrizationTwoLayerNeuralNet(input_dimension, output_dimension, overparametrization=1).to(device)
    loss_fn = torch.nn.MSELoss()
    
    freshly_initialized_model_loss = test(dataloader, model, loss_fn, device)
    
    with torch.no_grad():
        model.linear1.weight.copy_(torch.Tensor(numpy.identity(50)))
        model.linear2.weight.copy_(torch.Tensor(true_output_weights))
    
    true_model_loss = test(dataloader, model, loss_fn, device)
    assert 0 == int(true_model_loss) <= int(freshly_initialized_model_loss)

def execute_experiment(seed, input_dimension, output_dimension, sample_size, batch_size, epochs, learning_rate, 
                       models_overparametrizations, runs_per_model, save_models_path=None, save_experiments_path=None, 
                       plot_results=False, plot_results_on_canvas=None, verbose=False):
    device = initialize(seed)
    assertion(input_dimension, output_dimension, sample_size, batch_size, device)
            
    train_loader, true_output_weights = fresh_dataset_dataloader(input_dimension, sample_size, batch_size)
    test_loader = fresh_dataset_dataloader(input_dimension, sample_size, batch_size, true_output_weights)
    
    experiment = {
        'seed': seed,
        'input_dimension': input_dimension,
        'output_dimension': output_dimension,
        'sample_size': sample_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'distinction': 'overparametrization',
        'true_output_weights': true_output_weights.tolist(),
        'models_overparametrizations': models_overparametrizations,
        'models_runs': []
    }

    if save_models_path is not None: from utils.persistance import save_model
    if save_experiments_path is not None: from utils.persistance import save_experiment
    if plot_results or plot_results_on_canvas: from utils.plotting import plot_experiment
    
    for overparametrization in models_overparametrizations:
        if verbose: print(f'x{overparametrization} Oveparametrization')
    
        model_runs = []
        for run_number in range(runs_per_model):
            if verbose: print(f'Run {run_number}')

            model = OverparametrizationTwoLayerNeuralNet(overparametrization=overparametrization, **experiment).to(device)
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            run = {
                'train': [test(train_loader, model, loss_fn, device)],
                'train_time': [0],
                'test': [test(test_loader, model, loss_fn, device, verbose=verbose)]
            }
            
            for epoch in range(1, epochs + 1):
                if verbose: print(f'Epoch {epoch}')
                    
                start_time = time.time()
                train_loss = train(train_loader, model, loss_fn, optimizer, device, verbose=verbose)
                end_time = time.time()
                
                train_time = run['train_time'][-1] + end_time - start_time
                test_loss = test(test_loader, model, loss_fn, device, verbose=verbose)
                run['train'].append(train_loss), run['train_time'].append(train_time), run['test'].append(test_loss)

                if plot_results_on_canvas: 
                    intermidiate_result = {**experiment, 'models_runs': [*experiment['models_runs'], 
                        {'distinction': overparametrization, 'model_runs': [*model_runs, run]}]}
                        
                    plot_experiment(intermidiate_result, plot_results_on_canvas)
            
            model_runs.append(run)
            if save_models_path is not None: 
                model_parameters = {**experiment, 'run': run_number, 'overparametrization': overparametrization}
                save_model(model, model_parameters, save_models_path, MODEL_NAME_PARAMETERS)
    
        experiment['models_runs'].append({'distinction': overparametrization, 'model_runs': model_runs})
    
    if save_experiments_path is not None: save_experiment(experiment, save_experiments_path, EXPERIMENT_NAME_PARAMETERS)
    if plot_results: plot_experiment(experiment)
    return experiment

    
class OverparametrizationTwoLayerNeuralNet(torch.nn.Module):

    def __init__(self, input_dimension:int, output_dimension:int, overparametrization:int, *args, **kwargs):
        super(OverparametrizationTwoLayerNeuralNet, self).__init__()
        hidden_units = int(input_dimension * overparametrization)

        self.linear1 = torch.nn.Linear(input_dimension, hidden_units, bias=False)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_units, output_dimension, bias=False)
        

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
