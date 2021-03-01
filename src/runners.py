from tqdm import trange
from src import metrics
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter


def train(
    net,
    criterion,
    optimizer,
    lr_scheduler,
    train_dataloader,
    test_dataloader,
    n_epochs,
    device,
    save_path,
    save_every_n_epochs,
):
    """Trains the model on the supplied train data during n_epochs.

    Args:
        net (torch.nn.Module): model to train. normally defined in the models module
        criterion (torch.nn.modules.loss.Module): loss function
        optimizer (torch.optim): optimizer function
        lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler
        train_dataloader (torch.utils.data.dataloader.DataLoader): training set
        test_dataloader (torch.utils.data.dataloader.DataLoader): test set
        n_epochs (int): number of epochs to train
        device (str): 'cuda' or 'cpu'
        save_path (str): folder where the checkpoints of the models will be dumped
        save_every_n_epochs (int): number of epochs after checkpoint

    Returns:
        model (torch.nn.Module): model trained
    """
    log_dir = os.path.join(save_path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sw = SummaryWriter(log_dir=log_dir)
    net = net.to(device)
    pb = trange(n_epochs + 1, desc="", leave=True)
    # Calculate initial loss

    for epoch in pb:
        net.train()
        if epoch % save_every_n_epochs == 0:
            torch.save(
                {
                    "alias": os.path.split(save_path)[-1],
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(save_path, f"e_{epoch:03d}.pt"),
            )
        # Log train metrics
        metrics_dict_train = eval_epoch(
            net=net,
            dataloader=train_dataloader,
            device=device,
            metric_names=["crossentropy", "accuracy"],
        )
        metrics_train = [
            name + ": " + str(round(value, 3))
            for (name, value) in metrics_dict_train.items()
        ]
        avg_loss = metrics_dict_train["crossentropy"]
        avg_acc = metrics_dict_train["accuracy"]
        sw.add_scalar("train/crossentropy", avg_loss, epoch)
        sw.add_scalar("train/accuracy", avg_acc, epoch)

        # Log test metrics
        metrics_dict_test = eval_epoch(
            net=net,
            dataloader=test_dataloader,
            device=device,
            metric_names=["crossentropy", "accuracy"],
        )
        metrics_test = [
            name + ": " + str(round(value, 3))
            for (name, value) in metrics_dict_test.items()
        ]
        avg_loss_test = metrics_dict_test["crossentropy"]
        avg_acc_test = metrics_dict_test["accuracy"]
        sw.add_scalar("test/crossentropy", avg_loss_test, epoch)
        sw.add_scalar("test/accuracy", avg_acc_test, epoch)

        # Build the progress bar description
        pb.set_description(
            f"[EPOCH: {epoch}] [LR: {round(lr_scheduler.get_last_lr()[0], 8)}] Train_{' | Train_'.join(metrics_train)} | Test_{' | Test_'.join(metrics_test)}"
        )

        avg_loss = train_epoch(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
        )
        lr_scheduler.step()

    return net


def train_epoch(net, criterion, optimizer, dataloader, device):
    avg_loss = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss = train_step(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            inputs=inputs,
            labels=labels,
        )
        loss = loss.cpu().item()
        avg_loss += loss
    avg_loss /= i + 1
    return avg_loss


def eval_epoch(net, dataloader, device, metric_names=("accuracy",)):
    y_true = []
    y_pred = []
    metrics_dict = {}
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        y_true.append(labels.cpu().data.numpy())
        y_pred.append(outputs.cpu().data.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    for metric_name in metric_names:
        f = getattr(metrics, metric_name)
        metrics_dict[metric_name] = f(y_true=y_true, y_pred=y_pred)
    return metrics_dict


def train_step(net, criterion, optimizer, inputs, labels):
    # zero the parameter gradients
    net.train()  # Network in train mode
    optimizer.zero_grad()

    # Forward prop
    outputs = net(inputs)
    loss = criterion(outputs, labels)

    # Backward prop
    loss.backward()

    # Train step
    optimizer.step()

    return loss
