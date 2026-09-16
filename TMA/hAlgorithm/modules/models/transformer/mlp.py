import torch.nn as nn


class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        from third_party.rational_kat_cu.kat_rational import KAT_Group

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode=act_cfg["act_init"][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode=act_cfg["act_init"][1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import torch

    # Test configurations
    test_configs = [
        # Basic configuration with identity and gelu
        dict(
            in_features=256,
            hidden_features=512,
            out_features=256,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=True,
            drop=0.1,
        ),
        # Different activation combination
        dict(
            in_features=256,
            hidden_features=512,
            out_features=128,
            act_cfg=dict(type="KAT", act_init=["gelu", "identity"]),
            bias=True,
            drop=0.0,
        ),
    ]

    print("=== Testing KAN (Knowledge-Augmented Network) ===")

    for i, config in enumerate(test_configs, 1):
        print(f"\nTest Case {i}:")
        print(f"Configuration: {config}")

        # Create model
        model = KAN(**config).cuda()
        model.eval()  # Set to evaluation mode

        # Create test input
        batch_size = 4
        seq_length = 16
        x = torch.randn(batch_size, seq_length, config["in_features"]).cuda()

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Print results
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output features: {config['out_features']}")

        # Verify output dimensions
        assert output.shape == (
            batch_size,
            seq_length,
            config["out_features"],
        ), f"Output shape mismatch! Expected {(batch_size, seq_length, config['out_features'])}, got {output.shape}"

        # Basic numerical checks
        print("Output statistics:")
        print(f"- Mean: {output.mean().item():.4f}")
        print(f"- Std: {output.std().item():.4f}")
        print(f"- Min: {output.min().item():.4f}")
        print(f"- Max: {output.max().item():.4f}")

    print("\nAll tests passed successfully!")
