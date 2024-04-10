"""Evaluation utilities."""

import torch


def compute_accuracy(model, data_loader, device):
    """Compute the accuracy of the model on a dataset."""
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        _, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples
