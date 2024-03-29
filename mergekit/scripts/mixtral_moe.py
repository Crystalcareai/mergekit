import logging
import os
import sys
from typing import Dict, List, Optional, Union

import click
import re
import torch
import tqdm
import transformers
import yaml
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from mergekit.common import ModelReference, dtype_from_name
from mergekit.io import LazyTensorLoader, TensorWriter
from mergekit.merge import MergeOptions
from mergekit.options import add_merge_options


class Expert(BaseModel):
    source_model: str
    tokenizer: Optional[str] = None
    positive_prompts: List[str]
    negative_prompts: Optional[List[str]] = None
    noise_scale: Optional[float] = None

    @property
    def model_ref(self):
        return ModelReference.parse(self.source_model)


class MistralMOEConfig(BaseModel):
    experts: List[Expert]
    gate_mode: str = "hidden"
    dtype: Optional[str] = None
    experts_per_token: int = 2


def get_hidden_states(
    model: AutoModelForCausalLM,
    tokenized: transformers.BatchEncoding,
    average: bool = True,
) -> List[torch.Tensor]:
    with torch.no_grad():
        output: CausalLMOutputWithPast = model(
            **tokenized.to(model.device), output_hidden_states=True, return_dict=True
        )
    hidden_states = torch.stack(
        output.hidden_states[:-1]
    )  # (num_layers, batch_size, seq_len, hidden_size)
    if average:
        # use average over sequence
        hidden_states = hidden_states.sum(dim=2) / hidden_states.shape[2]
    else:
        # take last value
        hidden_states = hidden_states[:, :, -1, :]
    return hidden_states.sum(dim=1) / hidden_states.shape[1]


def get_cheap_embedding(
    embed: torch.Tensor,
    tokenized: Dict[str, torch.Tensor],
    num_layers: int,
    vocab_size: int,
) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(
        tokenized["input_ids"], num_classes=vocab_size
    )  # (batch_size, seq_len, 32000)
    h = onehot.float() @ embed.float()  # (batch_size, seq_len, hidden_size)
    embedded = (
        (h * tokenized["attention_mask"].unsqueeze(-1))
        .sum(dim=1)
        .sum(dim=0, keepdim=True)
    )  # (1, hidden_size)
    res = embedded / embedded.norm(dim=-1, keepdim=True).clamp(
        min=1e-8
    )  # (1, hidden_size)
    return res.repeat(num_layers, 1)


def tokenize_prompts(
    prompts: List[str], tokenizer: PreTrainedTokenizerBase
):
    return tokenizer(
        [tokenizer.bos_token + p for p in prompts],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )


def get_gate_params(
    model_ref: ModelReference,
    tokenizer: PreTrainedTokenizerBase,
    experts: List[Expert],
    mode: str = "hidden",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    lazy_unpickle: bool = False,
    trust_remote_code: bool = False,
    device: str = "auto",
):
    gate_vecs = []
    _do_it = None

    model_cfg = model_ref.config(trust_remote_code=trust_remote_code)

    if mode == "random":
        return torch.randn(
            (model_cfg.num_hidden_layers, len(experts), model_cfg.hidden_size)
        )
    elif mode == "cheap_embed":
        embed = LazyTensorLoader(
            model_ref.tensor_index(), lazy_unpickle=lazy_unpickle
        ).get_tensor("model.embed_tokens.weight")

        def _do_it(tokenized):
            return get_cheap_embedding(
                embed,
                tokenized,
                num_layers=model_cfg.num_hidden_layers,
                vocab_size=model_cfg.vocab_size,
            )

    elif mode in ("hidden", "hidden_avg", "hidden_last"):
        model = AutoModelForCausalLM.from_pretrained(
            model_ref.model.path,
            revision=model_ref.model.revision,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
        )

        def _do_it(tokenized):
            return get_hidden_states(
                model, tokenized=tokenized, average=mode == "hidden_avg"
            )

    gate_vecs = []
    for expert in tqdm.tqdm(experts, desc="expert prompts"):
        hidden_states = _do_it(tokenize_prompts(expert.positive_prompts, tokenizer))
        if expert.negative_prompts:
            hidden_states -= _do_it(
                tokenize_prompts(expert.negative_prompts, tokenizer)
            )

        hidden_states /= hidden_states.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        gate_vecs.append(hidden_states)
    gate_vecs = torch.stack(gate_vecs, dim=0)  # (num_expert, num_layer, hidden_size)
    return gate_vecs.permute(1, 0, 2)


def warn_degenerate_gates(gate_vecs: torch.Tensor, threshold: float = 5.0):
    degen_indices = []
    num_layers, _num_experts, _hidden_size = gate_vecs.shape
    for idx in range(num_layers):
        c = torch.linalg.cond(gate_vecs[idx, :, :].float())
        if c > threshold:
            degen_indices.append(idx)

    if degen_indices:
        if len(degen_indices) == 1:
            layer_str = f"layer {degen_indices[0]}"
            verb = "has"
        elif len(degen_indices) == 2:
            layer_str = f"layers {' and '.join(map(str, degen_indices))}"
            verb = "have"
        elif len(degen_indices) >= num_layers:
            layer_str = "ALL layers"
            verb = "have"
        else:
            layer_str = (
                "layers "
                + ", ".join(map(str, degen_indices[:-1]))
                + ", and "
                + str(degen_indices[-1])
            )
            verb = "have"

        logging.warning(
            f"{layer_str} {verb} degenerate routing parameters "
            "- your prompts may be too similar."
        )
        logging.warning("One or more experts will be underutilized in your model.")


def is_bad_config(config: MistralMOEConfig, allow_all_same: bool = False) -> bool:
    if len(config.experts) < 2:
        logging.error("Must include at least two experts.")
        return True

    if config.gate_mode == "random":
        return False  # eh we're good

    def prompt_tup(e: Expert):
        return (tuple(e.positive_prompts), tuple(e.negative_prompts or []))

    # let's just nip this trend in the bud
    p_first = prompt_tup(config.experts[0])
    if all(prompt_tup(e) == p_first for e in config.experts[1:]):
        logging.error(
            "Your positive and negative prompts are identical for all experts. This will not produce a functioning MoE."
        )
        logging.error(
            "For each expert, `positive_prompts` must contain one or more example prompt reflecting what should be routed to that expert."
        )
        return True

    if not allow_all_same:
        if all(
            e.source_model == config.experts[0].source_model for e in config.experts[1:]
        ):
            logging.error(
                "All of your expert models are the same. This will produce "
                "a model that uses more resources but gives the exact same output. "
                "If you plan to train the model after merging, proceed with the "
                "--i-understand-this-is-not-useful-without-training flag."
            )
            return True


def get_parametered_layers_list(model, num_hidden_layers):
    """
    Given a model and the number of hidden layers, this function returns two lists:
    one containing the parameterized layers of the model, grouped by hidden layer,
    and the other containing the remaining parameterized layers of the model.
    """
    layers = []
    for name, module in model.named_parameters():
        layers.append((name, module))
    hidden_layers = [[] for _ in range(num_hidden_layers)]

    other_layers = []

    for layer in layers:
        if re.search(r"layers\.\d+", layer[0]):
            hidden_layers[int(layer[0].split("layers.")[1].split(".")[0])].append(layer)
        else:
            other_layers.append(layer)

    return hidden_layers, other_layers


def copy_weight(model, moe_model, expert_index=None):
    """
    Copy weights from model to moe_model for matching layers.
    If expert_index is provided, copy weights to the specified expert in the MoE model.
    """
    model_hidden_layers, model_other_layers = get_parametered_layers_list(model, model.config.num_hidden_layers)
    moe_hidden_layers, moe_other_layers = get_parametered_layers_list(moe_model, moe_model.config.num_hidden_layers)

    model_other_layers = {layer[0]: layer[1] for layer in model_other_layers}
    moe_other_layers = {layer[0]: layer[1] for layer in moe_other_layers}

    if expert_index is None:
        # Copy embeddings and other layers from the first expert
        for other_layer in model_other_layers:
            if other_layer in moe_other_layers:
                moe_other_layers[other_layer].data = model_other_layers[other_layer].data
            else:
                logging.warning(
                    f"Layer {other_layer} not found in MoE model. Skipping."
                )
    else:
        # Copy expert model weights to the corresponding expert in the MoE model
        num_model_layers = len(model_hidden_layers)
        num_moe_layers = len(moe_hidden_layers)

        if num_model_layers != num_moe_layers:
            logging.warning(
                f"Expert model {expert_index} has {num_model_layers} hidden layers, "
                f"while the MoE model has {num_moe_layers} hidden layers. "
                "Copying weights to matching layers."
            )

        num_layers = min(num_model_layers, num_moe_layers)

        for i in range(num_layers):
            model_layer = model_hidden_layers[i]
            moe_layer = moe_hidden_layers[i]

            for weight_name, weight in model_layer:
                if "mlp" not in weight_name:
                    continue
                
                moe_weight_name = weight_name.replace(".mlp.", f".block_sparse_moe.experts.{expert_index}.")
                moe_weight = next((w for n, w in moe_layer if n == moe_weight_name), None)
                if moe_weight is not None:
                    moe_weight.data.copy_(weight.data)
                else:
                    print(f"Warning: Weight {moe_weight_name} not found in MoE layer.")
def build(
    config: MistralMOEConfig,
    out_path: str,
    merge_options: MergeOptions,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device: str = "auto",
    allow_all_same: bool = False,
):
    if is_bad_config(config, allow_all_same=allow_all_same):
        sys.exit(1)

    if config.experts_per_token < 1:
        logging.error("Experts per token must be >= 1")
        sys.exit(1)
    if config.experts_per_token > len(config.experts):
        logging.error("Experts per token must be <= number of experts")
        sys.exit(1)

    expert_models = []
    expert_tokenizers = []
    expert_configs = []

    for expert in config.experts:
        expert_model = AutoModelForCausalLM.from_pretrained(expert.source_model)
        expert_tokenizer = AutoTokenizer.from_pretrained(expert.tokenizer or expert.source_model)
        expert_config = expert_model.config

        expert_models.append(expert_model)
        expert_tokenizers.append(expert_tokenizer)
        expert_configs.append(expert_config)

    out_cfg = AutoConfig.from_pretrained(config.experts[0].source_model)
    out_cfg.architectures = ["MixtralForCausalLM"]
    out_cfg.num_local_experts = len(config.experts)
    out_cfg.num_experts_per_tok = config.experts_per_token
    out_cfg.sliding_window = None
    if config.dtype:
        out_cfg.torch_dtype = config.dtype
    out_cfg.save_pretrained(out_path)

    out_model = AutoModelForCausalLM.from_config(out_cfg)

    if (out_cfg.num_local_experts & (out_cfg.num_local_experts - 1)) != 0:
        logging.warning(
            f"Your model has {out_cfg.num_local_experts} experts, which is "
            "not a power of two. The model will not be usable in llama.cpp."
        )

    writer = TensorWriter(
        out_path=out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )

    if config.dtype:
        out_dtype = dtype_from_name(config.dtype)
    elif expert_configs[0].torch_dtype:
        out_dtype = expert_configs[0].torch_dtype
        if isinstance(out_dtype, str):
            out_dtype = dtype_from_name(out_dtype)
    else:
        out_dtype = None

    logging.info("Copying parameters...")

    # Copy embeddings and other layers from the first expert
    copy_weight(expert_models[0], out_model)

    for i, expert_model in enumerate(expert_models):
        copy_weight(expert_model, out_model, expert_index=i)

    tokenizer = expert_tokenizers[0]
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.bos_token_id

    logging.info("Getting gate parameters...")
    gate_vecs = get_gate_params(
        ModelReference.parse(config.experts[0].source_model),
        tokenizer,
        config.experts,
        mode=config.gate_mode,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        lazy_unpickle=merge_options.lazy_unpickle,
        trust_remote_code=merge_options.trust_remote_code,
        device=device,
)
    warn_degenerate_gates(gate_vecs)

    for layer_idx in range(out_cfg.num_hidden_layers):
        writer.save_tensor(
            f"model.layers.{layer_idx}.block_sparse_moe.gate.weight",
            gate_vecs[layer_idx, :, :].contiguous().to(dtype=out_dtype),
        )
    writer.finalize()

    if merge_options.copy_tokenizer:
        logging.info("Saving tokenizer...")
        tokenizer.save_pretrained(out_path, safe_serialization=True)

    logging.info("Done.")

@click.command("mergekit-moe")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path())
@click.option(
    "--load-in-4bit",
    is_flag=True,
    type=bool,
    default=False,
    help="Load model in 4bit for computing hidden states",
)
@click.option(
    "--load-in-8bit",
    is_flag=True,
    type=bool,
    default=False,
    help="Load model in 8bit for computing hidden states",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to use to compute embeddings",
    show_default=True,
)
@click.option(
    "--verbose", "-v", type=bool, default=False, is_flag=True, help="Verbose logging"
)
@click.option(
    "--i-understand-this-is-not-useful-without-training",
    type=bool,
    default=False,
    is_flag=True,
    help="Really make the questionable model you want.",
)
@add_merge_options
def main(
    config_path: str,
    out_path: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    device: str,
    merge_options: MergeOptions,
    verbose: bool,
    i_understand_this_is_not_useful_without_training: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    if merge_options.cuda:
        logging.warning(
            '--cuda is a no-op for mergekit-moe, use "--device cuda" instead'
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config_source = file.read()

    config = MistralMOEConfig.model_validate(yaml.safe_load(config_source))
    build(
        config,
        out_path=out_path,
        merge_options=merge_options,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device=device,
        allow_all_same=i_understand_this_is_not_useful_without_training,
    )

    if merge_options.write_model_card:
        # TODO: generate a README.md as well
        with open(
            os.path.join(out_path, "mergekit_moe_config.yml"), "w", encoding="utf-8"
        ) as fp:
            fp.write(config_source)

if __name__ == "__main__":
    main()
