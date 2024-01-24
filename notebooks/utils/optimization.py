import numpy, torch

def initialize(seed=123):
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def train(dataloader, model, loss_fn, optimizer, device, verbose=False):
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

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
    with torch.no_grad():
        test_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    if verbose:
        print(f"Test Avg loss: {test_loss:>8f}\n")
        
    return test_loss