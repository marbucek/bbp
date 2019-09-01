import torch
import torch.nn.functional as F
import numpy as np

class RMSE():

    def __init__(self):
        self.__name__ = 'RMSE'

    def __call__(self, output, target, output_column = 0):
        if output.dim() > 1:
            output = output[:,output_column]
        return float(torch.sqrt(F.mse_loss(output, target)))

def correlation(output, target, output_column = 0):
    if output.dim() > 1:
        output = output[:,output_column]
    vx = output - output.mean()
    vy = target - target.mean()
    corr = torch.sum(vx*vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return float(corr)

def accuracy(output, target):
    pred = torch.argmax(output, dim = 1)
    no_correct = np.sum(pred.cpu().detach().numpy() == target.cpu().detach().numpy())
    return no_correct/len(output)
