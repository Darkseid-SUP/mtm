import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, criterion, train, test=None, validation=None, epochs=10, minibatch=float('inf'), temp_file_path=None, patience=1, print_every=1, lr=0.001, weight_decay=0, optimizer_type="adam", lr_modifiers=None, **kwargs):
    model_last = model

    out = []
    for rs in range(len(lr)):
        model = model_last

        if lr_modifiers is None:
            params = model.parameters()
        else:
            if len(lr_modifiers) == len(model.parameters()):
                params = [{"params": model.parameters()[i], "lr": lr_modifiers[i] * lr[rs]} for i in range(len(model.parameters()))]
            else:
                raise ValueError("invalid lr_modifiers")

        optimizer = {
            "adam": optim.Adam(params, lr=lr[rs], weight_decay=weight_decay),
            "sgd": optim.SGD(params, lr=lr[rs], weight_decay=weight_decay),
            "adadelta": optim.Adadelta(params, lr=lr[rs], weight_decay=weight_decay),
            "asgd": optim.ASGD(params, lr=lr[rs], weight_decay=weight_decay),
            "lbfgs": optim.LBFGS(params, lr=lr[rs]),
            "rmsprop": optim.RMSprop(params, lr=lr[rs], weight_decay=weight_decay),
            "rprop": optim.Rprop(params, lr=lr[rs])
        }[optimizer_type]

        if isinstance(minibatch, int):
            minibatch_sampler = DataLoader(train, batch_size=minibatch, shuffle=True)
        else:
            minibatch_sampler = minibatch

        log = {"epoch": list(range(1, epochs+1)),
               "loss_train": [float('inf')]*epochs,
               "loss_test": [float('inf')]*epochs,
               "loss_validation": [float('inf')]*epochs}

        for e in range(1, epochs+1):
            mbs = minibatch_sampler

            if e > 1:
                model.train()
                for mb in mbs:
                    optimizer.zero_grad()
                    inputs, targets = mb
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            if test is not None:
                model.eval()
                with torch.no_grad():
                    inputs, targets = test
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    log["loss_test"][e-1] = loss.item()

            if validation is not None:
                model.eval()
                with torch.no_grad():
                    inputs, targets = validation
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    log["loss_validation"][e-1] = loss.item()

            if e % print_every == 0:
                print(f"Epoch {e}/{epochs} - Train Loss: {log['loss_train'][e-1]:.4f} - Test Loss: {log['loss_test'][e-1]:.4f} - Validation Loss: {log['loss_validation'][e-1]:.4f}")

        out.append(log)

    return out