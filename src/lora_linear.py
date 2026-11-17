import torch
import copy
import json
from torch import nn
from pathlib import Path

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank, alpha):
        super().__init__()
        
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("rank must be a positive integer")

        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        self.in_features = self.base.in_features
        self.out_features = self.base.out_features

        self.A = nn.Parameter(
            self.base.weight.new_empty((rank, self.in_features))
        )
        self.B = nn.Parameter(
            self.base.weight.new_zeros((self.out_features, rank))
        )

        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.zeros_(self.B)






    def forward(self, x):
        base_output = self.base(x)

        low_rank = x @ self.A.T
        lora_output = low_rank @ self.B.T

        return base_output + self.scaling * lora_output

    def to_merged_linear(self):
        merged = copy.deepcopy(self.base)

        with torch.no_grad():
            update = self.scaling * (self.B @ self.A)
            merged.weight.copy_(self.base.weight + update)

        return merged
    



if __name__ == "__main__":
    torch.manual_seed(42)
    base = nn.Linear(3, 2)
    layer = LoRALinear(base, rank=1, alpha=1)

    print("Input features:", layer.in_features)
    print("Output features:", layer.out_features)
    print("Scaling:", layer.scaling)

    for name, parameter in layer.named_parameters():
        print(name, tuple(parameter.shape), parameter.requires_grad)
    print("A:", layer.A)
    print("B:", layer.B)
    print("Initial update:", layer.B @ layer.A)

    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    base_output = base(x)
    lora_output = layer(x)

    print("\nBase output:", base_output)
    print("LoRA output:", lora_output)

    torch.testing.assert_close(
        lora_output, base_output, rtol=1e-5, atol=1e-6
    )
    print("Initial output equivalence: PASS")

    initial_update = layer.scaling * (layer.B @ layer.A)

    torch.testing.assert_close(
        initial_update,
        torch.zeros_like(initial_update),
        rtol=0,
        atol=1e-7,
    )

    initial_output_max_diff = (
        lora_output - base_output
    ).abs().max().item()


    # Train the LoRA layer
    target = torch.tensor([
        [1.0, 0.0],
        [2.0, 1.0],
    ])

    loss = nn.MSELoss()(layer(x), target)
    loss.backward()

    print("\nLoss:", loss.item())
    print("Base weight gradient:", layer.base.weight.grad)
    print("Base bias gradient:", layer.base.bias.grad)
    print("A gradient:", layer.A.grad)
    print("B gradient:", layer.B.grad)

    optimizer = torch.optim.SGD(
        [layer.A, layer.B], lr=0.01
    )

    optimizer.step()

    print("\nB after first update:", layer.B)

    optimizer.zero_grad(set_to_none=True)

    second_loss = nn.MSELoss()(layer(x), target)
    second_loss.backward()

    print("Second loss:", second_loss.item())
    print("A gradient after B update:", layer.A.grad)
    print("B gradient after B update:", layer.B.grad)

    base_weight_before = layer.base.weight.detach().clone()
    base_bias_before = layer.base.bias.detach().clone()
    A_before = layer.A.detach().clone()
    B_before = layer.B.detach().clone()

    optimizer.step()

    assert torch.equal(layer.base.weight, base_weight_before)
    assert torch.equal(layer.base.bias, base_bias_before)

    print("\nFrozen base verification: PASS")
    print("A changed:", not torch.equal(layer.A, A_before))
    print("B changed:", not torch.equal(layer.B, B_before))
    assert layer.A.grad is not None, "A gradient was not computed"
    assert layer.B.grad is not None, "B gradient was not computed"
    assert layer.A.shape == (layer.rank, layer.in_features)
    assert layer.B.shape == (layer.out_features, layer.rank)

    expected_count = layer.rank * (
        layer.in_features + layer.out_features
    )
    actual_count = sum(
        p.numel() for p in layer.parameters() if p.requires_grad
    )

    assert actual_count == expected_count
    print("Trainable parameter count:", actual_count)



    results = {
        "initial_output_equivalence_passed": True,
        "initial_update_zero_passed": True,
        "initial_output_max_abs_difference": initial_output_max_diff,
        "A_shape": list(layer.A.shape),
        "B_shape": list(layer.B.shape),
        "expected_trainable_parameters": expected_count,
        "actual_trainable_parameters": actual_count,
        "experiment": "educational_lora_gradient_check",
        "seed": 42,
        "pytorch_version": torch.__version__,
        "device": str(layer.A.device),
        "rank": layer.rank,
        "alpha": layer.alpha,
        "optimizer": "SGD",
        "learning_rate": optimizer.param_groups[0]["lr"],
        "base_weight_unchanged": torch.equal(
            layer.base.weight, base_weight_before
        ),
        "base_bias_unchanged": torch.equal(
            layer.base.bias, base_bias_before
        ),
        "A_changed": not torch.equal(layer.A, A_before),
        "B_changed": not torch.equal(layer.B, B_before),
        "A_gradient_norm_before_second_step": layer.A.grad.norm().item(),
        "B_gradient_norm_before_second_step": layer.B.grad.norm().item(),
        "loss_before_first_step": loss.item(),
        "loss_after_first_step": second_loss.item(),
    }

    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs/lora_gradient_check.json")
    log_path.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("Log saved to:", log_path)



