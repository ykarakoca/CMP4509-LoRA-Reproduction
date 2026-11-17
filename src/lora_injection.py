from torch import nn

from lora_linear import LoRALinear


def inject_roberta_lora(model, rank=8, alpha=8):
    targets = []

    for layer_index, layer in enumerate(model.roberta.encoder.layer):
        attention = layer.attention.self

        for name in ("query", "value"):
            projection = getattr(attention, name)

            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"Layer {layer_index} {name}: expected nn.Linear, "
                    f"found {type(projection).__name__}"
                )

            targets.append((attention, name, projection))

    for attention, name, projection in targets:
        setattr(attention, name, LoRALinear(projection, rank=rank, alpha=alpha))

    return len(targets)