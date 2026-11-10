
import torch

from mnist_nmf.models import MNISTMLP


def main():
    # Create a batch of 64 random MNIST images
    x = torch.randn(64, 1, 28, 28)

    # Build ReLU and Tanh models
    relu_model = MNISTMLP(activation="relu")
    tanh_model = MNISTMLP(activation="tanh")

    # Extract hidden-layer activations
    relu_features = relu_model.forward_features(x)
    tanh_features = tanh_model.forward_features(x)

    # Obtain classification logits
    relu_logits = relu_model(x)
    tanh_logits = tanh_model(x)

    print("===== ReLU Model =====")
    print("Hidden shape:", relu_features.shape)
    print("Output shape:", relu_logits.shape)
    print("Minimum activation:", relu_features.min().item())
    print("Maximum activation:", relu_features.max().item())
    print(
    "Zero ratio:",
    (relu_features == 0).float().mean().item()
)

    print("===== Tanh Model =====")
    print("Hidden shape:", tanh_features.shape)
    print("Output shape:", tanh_logits.shape)
    print("Minimum activation:", tanh_features.min().item())
    print("Maximum activation:", tanh_features.max().item())
    print(
    "Negative ratio:",
    (tanh_features < 0).float().mean().item()
)

    # Count model parameters
    num_params = sum(
        p.numel() for p in relu_model.parameters()
        if p.requires_grad
    )

    print("Total trainable parameters:", num_params)


if __name__ == "__main__":
    main()
