import torch
import numpy as np
from time import time

def epoch_pass(model, loader, metrics, device, train = True, lossfun = None, optimizer = None, scheduler = None, return_output = False):

    if train:
        model.train()
        if scheduler != None:
            scheduler.step()
    else:
        model.eval()

    output_all = []; target_all = []; total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):

        output = model(data)
        output_all.append(output); target_all.append(target)
        batch_loss = lossfun(output, target)*len(data)
        total_loss += batch_loss

        if train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    output_all = torch.cat(output_all, dim = 0)
    target_all = torch.cat(target_all, dim = 0)

    metrics_dict = {}
    metrics_dict['loss'] = float(total_loss/len(loader.dataset))
    for metric in metrics:
        output, target = output_all.cpu(), target_all.cpu()
        metrics_dict[metric.__name__] = metric(output_all, target_all)

    if return_output == False:
        return metrics_dict
    else:
        return metrics_dict, output_all, target_all

def train(epochs, model, train_loader, valid_loader, metrics, device, lossfun, optimizer, scheduler, history = False):

    best_score = np.infty
    train_history = {}; valid_history = {}

    for epoch in range(epochs):
        out_train = epoch_pass(model, train_loader, metrics,
                device = device, train = True, lossfun = lossfun, optimizer = optimizer, scheduler= scheduler)
        out_train['epoch'] = epoch
        if scheduler != None:
            out_train['lr'] = scheduler.get_lr()[0]

        out_valid = epoch_pass(model, valid_loader, metrics,
                device = device, train = False, lossfun = lossfun)
        out_valid['epoch'] = epoch

        if history:
            if epoch == 0:
                for key in out_train.keys():
                    train_history[key] = [out_train[key]]
                for key in out_valid.keys():
                    valid_history[key] = [out_valid[key]]
            else:
                for key in out_train.keys():
                    train_history[key].append(out_train[key])
                for key in out_valid.keys():
                    valid_history[key].append(out_valid[key])

    state_dict = model.state_dict()

    if history:
        return out_valid, state_dict, train_history, valid_history
    else:
        return out_valid, state_dict
