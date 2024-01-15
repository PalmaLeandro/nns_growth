import matplotlib.pyplot

def plot_experiment(experiment_specification):
    sample_size = experiment_specification['sample_size']
    epochs = experiment_specification['epochs']
    models_runs = experiment_specification['models_runs']
    
    models_overparametrization = sorted(list(set([model_run['overparametrization'] for model_run in models_runs])))
    different_models = len(models_overparametrization)
    
    fig, models_axes = matplotlib.pyplot.subplots(different_models, 2, figsize=(16, different_models * 8))
    
    for model_overparameterization, (model_train_ax, model_test_ax) in zip(models_overparametrization, models_axes):
        model_runs = [
            model_run for model_run in models_runs if model_run['overparametrization'] == model_overparameterization
        ][0]['model_runs']
        
        model_train_ax.set_title(f'x{model_overparameterization} Oveparametrization Train Loss')
        model_train_ax.set_xlabel('SGD iterations')
        model_train_ax.set_ylabel('Mean Squared Error')
        model_train_ax.set_xlim(0, sample_size * epochs)
        model_train_ax.set_ylim(0, 10)
        model_train_ax.grid(zorder=0)
        
        model_test_ax.set_title(f'x{model_overparameterization} Overparametrization Test Loss')
        model_test_ax.set_xlabel('SGD iterations')
        model_test_ax.set_ylabel('Mean Squared Error')
        model_test_ax.set_xlim(0, sample_size * epochs)
        model_test_ax.set_ylim(0, 10)
        model_test_ax.grid(zorder=0)

        min_train = min_test = 10
        for run, model_run in enumerate(model_runs):
            model_train_ax.plot([sample_size * epoch for epoch in range(epochs + 1)], model_run['train'], label=f'Run {run}')
            model_test_ax.plot([sample_size * epoch for epoch in range(epochs + 1)], model_run['test'], label=f'Run {run}')
            
            min_train = min(min_train, *model_run['train'])
            min_test = min(min_test, *model_run['test'])

        model_train_ax.hlines(min_train, 0, sample_size * epochs + 1, colors='grey', linestyles='dashed')
        model_test_ax.hlines(min_test, 0, sample_size * epochs + 1, colors='grey', linestyles='dashed')

        model_train_ax.text(0, min_train, f'{min_train:.2f}', horizontalalignment='right')
        model_test_ax.text(0, min_test, f'{min_test:.2f}', horizontalalignment='right')
    
        model_train_ax.legend()
        model_test_ax.legend()