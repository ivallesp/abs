from src.data import load_cifar10
from src.models import get_model
from src.runners import train

# Load data
train_dl, test_dl, classes = load_cifar10()

# Load model
net, criterion, optimizer = get_model("TestNet", {})

# Train and eval
train(
    net=net,
    criterion=criterion,
    optimizer=optimizer,
    train_dataloader=train_dl,
    test_dataloader=test_dl,
    n_epochs=100,
    device="cuda",
)
