
import torch
import torch.nn as nn


class MNISTMLP(nn.Module):
    """
    A simple MLP for MNIST classification.

    Architecture:
        Input (784)
        -> Hidden Layer (128)
        -> Activation (ReLU or Tanh)
        -> Output (10)
    """

    def __init__(self, activation="relu"):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(784, 128)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(
                "activation must be 'relu' or 'tanh'"
            )

        self.fc2 = nn.Linear(128, 10)

    def forward_features(self, x):
        """
        Extract hidden-layer activations.
        """
        x = self.flatten(x)
        x = self.fc1(x)
        h = self.activation(x)

        return h

    def forward(self, x):
        """
        Forward propagation for classification.
        """
        h = self.forward_features(x)
        logits = self.fc2(h)

        return logits
