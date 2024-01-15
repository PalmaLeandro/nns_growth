import numpy
import torch

def initialization(seed=123):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def fresh_dataset_dataloader(input_dimension, sample_size, output_weights):
    X = numpy.random.normal(size=(sample_size, input_dimension))
    y = numpy.matmul(X * (X > 0), output_weights)
    
    with torch.no_grad():
        tensor_X = torch.Tensor(X)
        tensor_y = torch.Tensor(y).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    dataloader = torch.utils.data.DataLoader(dataset)
    
    return dataloader

def train(dataloader, model, loss_fn, optimizer, device, verbose=False):
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()

    train_loss /= num_batches
    if verbose:
        print(f"Train Avg loss: {train_loss:>8f}")
        
    return train_loss

def test(dataloader, model, loss_fn, device, verbose=False):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    if verbose:
        print(f"Test Avg loss: {test_loss:>8f}\n")
        
    return test_loss

def assertion(input_dimension, output_dimension, sample_size, device):
    true_output_weights = numpy.random.choice([-1, 1], input_dimension)

    dataloader = fresh_dataset_dataloader(input_dimension, sample_size, true_output_weights)
    
    model = TwoLayerNeuralNet(
        input_dimension=input_dimension, 
        output_dimension=output_dimension, 
        hidden_units=input_dimension
    ).to(device)
    
    loss_fn = torch.nn.MSELoss()
    
    random_model_loss = test(dataloader, model, loss_fn, device)
    
    with torch.no_grad():
        model.linear1.weight.copy_(torch.Tensor(numpy.identity(50)))
        model.linear2.weight.copy_(torch.Tensor(true_output_weights))
    
    true_model_loss = test(dataloader, model, loss_fn, device)

    assert 0 == int(true_model_loss) <= int(random_model_loss)

def execute_experiment(seed, input_dimension, output_dimension, sample_size, models_overparametrization, 
                       runs_per_model, epochs, learning_rate):
    device = initialization(seed)
    assertion(input_dimension, output_dimension, sample_size, device)
    
    true_output_weights = numpy.random.choice([-1, 1], input_dimension)
    
    models_runs = []
    for model_overparameterization in models_overparametrization:
        print(f'x{model_overparameterization} Oveparametrization Model')
    
        model_runs = []
        for run in range(runs_per_model):
            print(f'Run {run}')
            train_loader = fresh_dataset_dataloader(input_dimension, sample_size, true_output_weights)
            test_loader = fresh_dataset_dataloader(input_dimension, sample_size, true_output_weights)
    
            model = TwoLayerNeuralNet(
                input_dimension=input_dimension, 
                output_dimension=output_dimension, 
                hidden_units=int(input_dimension * model_overparameterization)
            ).to(device)
    
            loss_fn = torch.nn.MSELoss()
    
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
            print('Freshly initialized model')
            train_losses = [test(train_loader, model, loss_fn, device)]
            test_losses = [test(test_loader, model, loss_fn, device, verbose=True)]
            
            for epoch in range(1, epochs + 1):
                print(f'Epoch {epoch}')
                train_loss = train(train_loader, model, loss_fn, optimizer, device, verbose=True)
                test_loss = test(test_loader, model, loss_fn, device, verbose=True)
    
                train_losses.append(train_loss)
                test_losses.append(test_loss)
            
            model_runs.append({'train': train_losses, 'test': test_losses})
    
        models_runs.append({'overparametrization': model_overparameterization, 'model_runs': model_runs})
    
    return {
        'seed': seed,
        'input_dimension': input_dimension,
        'output_dimension': output_dimension,
        'sample_size': sample_size,
        'models_overparametrization': models_overparametrization,
        'runs_per_model': runs_per_model,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'true_output_weights': true_output_weights.tolist(),
        'models_runs': models_runs
    }

    
class TwoLayerNeuralNet(torch.nn.Module):

    def __init__(self, input_dimension:int, output_dimension:int, hidden_units:int):
        super(TwoLayerNeuralNet, self).__init__()

        self.linear1 = torch.nn.Linear(input_dimension, hidden_units, bias=False)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_units, output_dimension, bias=False)
        

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
