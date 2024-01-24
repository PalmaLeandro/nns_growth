import io
import numpy
import matplotlib.pyplot

def plot_experiment(experiment_specification, canvas=None):
    sample_size = experiment_specification['sample_size']
    models_runs = experiment_specification['models_runs']
    batch_size = experiment_specification['batch_size']
    legend = experiment_specification.get('distinction', '')
    models_distinctions = sorted(list(set([model_run['distinction'] for model_run in models_runs])))
    
    if canvas: matplotlib.pyplot.ioff()
    fig, (model_train_ax, model_test_ax) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8))
        
    model_train_ax.set_title('Train Loss'), model_test_ax.set_title('Test Loss')
    x_axis_text = f'{"SGD (batch size = " + str(batch_size) + ")" if batch_size < sample_size else "GD"} iterations'
    model_train_ax.set_xlabel(x_axis_text), model_test_ax.set_xlabel(x_axis_text)

    model_train_ax.set_ylabel('Mean Squared Error'), model_test_ax.set_ylabel('Mean Squared Error')
    model_train_ax.set_yscale('log'), model_test_ax.set_yscale('log')

    train_y_scale_min, train_y_scale_max, test_y_scale_min, test_y_scale_max, max_epochs = 5, 0.1, 5, 0.1, 0
    
    for models_distinction in models_distinctions:
        model_runs = [
            model_run for model_run in models_runs if model_run['distinction'] == models_distinction
        ][0]['model_runs']
        train_losses = [model_run['train'] for model_run in model_runs]
        test_losses = [model_run['test'] for model_run in model_runs]
        epochs = max(*[len(model_run['train']) for model_run in model_runs], *[len(model_run['test']) for model_run in model_runs])

        train_losses = numpy.array([train_loss + ([] if len(train_loss) == epochs else ([numpy.nan] * (epochs - len(train_loss))))
                                    for train_loss in train_losses])
        test_losses = numpy.array([test_loss + ([] if len(test_loss) == epochs else ([numpy.nan] * (epochs - len(test_loss))))
                                   for test_loss in test_losses])

        avg_train_loss, std_train_loss = numpy.nanmean(train_losses, axis=0), numpy.nanstd(train_losses, axis=0)
        avg_test_loss, std_test_loss = numpy.nanmean(test_losses, axis=0), numpy.nanstd(test_losses, axis=0)

        iterations = [iteration * sample_size / batch_size for iteration in range(epochs)]

        model_train_ax.plot(iterations, avg_train_loss, label=f'{models_distinction}', zorder=3)
        model_train_ax.fill_between(iterations, avg_train_loss - std_train_loss, avg_train_loss + std_train_loss, zorder=2, alpha=0.2)
        
        model_test_ax.plot(iterations, avg_test_loss, label=f'{models_distinction}', zorder=3)
        model_test_ax.fill_between(iterations, avg_test_loss - std_test_loss, avg_test_loss + std_test_loss, zorder=2, alpha=0.2)
            
        min_train, max_train = avg_train_loss.min(), avg_train_loss.max()
        min_test, max_test = avg_test_loss.min(), avg_test_loss.max()
    
        model_train_ax.hlines(min_train, 0, float(epochs * sample_size / batch_size) + 1, colors='grey', linestyles='dashed')
        model_test_ax.hlines(min_test, 0, float(epochs * sample_size / batch_size) + 1, colors='grey', linestyles='dashed')
    
        model_train_ax.text(0, min_train, f'{min_train:.2f}', horizontalalignment='right')
        model_test_ax.text(0, min_test, f'{min_test:.2f}', horizontalalignment='right')

        max_epochs = max(max_epochs, epochs)
        train_y_scale_min, test_y_scale_min = min(train_y_scale_min, min_train), min(test_y_scale_min, min_test)
        train_y_scale_max, test_y_scale_max = max(train_y_scale_max, max_train), max(test_y_scale_max, max_test)
    
    model_train_ax.legend(title=legend), model_test_ax.legend(title=legend)

    model_train_ax.set_xlim(0, max_epochs * sample_size / batch_size), model_test_ax.set_xlim(0, max_epochs * sample_size / batch_size)
    model_train_ax.set_ylim(train_y_scale_min, train_y_scale_max), model_test_ax.set_ylim(test_y_scale_min, test_y_scale_max)

    ticks_args = dict(style='sci', axis='x', scilimits=(0,0))
    model_train_ax.ticklabel_format(**ticks_args), model_test_ax.ticklabel_format(**ticks_args)
    model_train_ax.xaxis.major.formatter._useMathText = model_test_ax.xaxis.major.formatter._useMathText = True

    if canvas: draw_figure_into_canvas(fig, canvas)

def draw_figure_into_canvas(figure, canvas):
    import ipywidgets

    try:
        buffer = io.BytesIO()
        figure.tight_layout()
        figure.savefig(buffer, format='png', bbox_inches='tight')
        matplotlib.pyplot.close(figure)
        canvas.draw_image(ipywidgets.Image(value=buffer.getvalue()), width=canvas.width, height=canvas.height)
    except:
        print('''ERROR: Couldn't print figure.''')
