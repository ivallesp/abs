import argparse
from src.data import get_dataset
from src.models import get_model
from src.runners import train
from src.model_tools import set_random_seed, initialize_weights
from src.paths import get_model_folder


N_EPOCHS = 100


ACTIVATIONS = [
    "absolute",
    "relu",
    "leaky_relu",
    "silu",
    "elu",
    "tanh",
    #"softplus",
    #"selu",
    #"gelu",
    #"sigmoid",
]


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--model-name",
        required=True,
        help="Name of the model to train.",
    )
    argparser.add_argument(
        "-d",
        "--dataset-name",
        required=True,
        help="Name of the dataset to train on.",
    )
    argparser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=True,
        help="Random seed to use along all the experiment.",
    )
    return argparser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    random_seed = args.seed

    # Set the random seed
    set_random_seed(random_seed)

    # Load data
    train_dl, test_dl, classes, input_size, input_channels = get_dataset(
        name=dataset_name, params={}
    )

    for activation_name in ACTIVATIONS:

        alias = f"{dataset_name}_{model_name}_{activation_name}_{random_seed}"
        model_folder = get_model_folder(alias)

        print(f"Training model: '{alias}'")

        # Load model
        net, criterion, optimizer, lr_scheduler = get_model(
            name=model_name,
            params={
                "n_outputs": len(classes),
                "activation_name": activation_name,
                "input_size": input_size,
                "input_channels": input_channels,
            },
            n_epochs=N_EPOCHS,
        )
        # initialize_weights(net)

        # Train and eval
        train(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dl,
            test_dataloader=test_dl,
            n_epochs=N_EPOCHS,
            device="cuda",
            save_path=model_folder,
            save_every_n_epochs=5,
        )


if __name__ == "__main__":
    main()