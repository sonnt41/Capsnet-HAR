import torch


def evaluation(output, label):
    rmax, predicted = torch.max(output, 1)
    total = label.shape[0]
    correct = (predicted == label).sum()
    accuracy = 100 * correct / total
    return correct, accuracy, predicted



def accuracy(output, label):
    rmax, predicted = torch.max(output, 1)
    total = label.shape[0]
    correct = (predicted == label).sum().data[0]
    accuracy = 100 * correct / total
    return correct, total
