import torch
import re


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    """Rounds up x to the next multiple of multiple_of."""
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


def convert_single_weights_to_fused_weights(
    single_weight_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    def get_matching_keys_for(regex: str, keys: list[str]):
        matching_keys = []
        for key in keys:
            if re.match(regex, key):
                matching_keys.append(key)
        return matching_keys

    def concat_weights_and_biases(
        state_dict: dict[str, torch.Tensor],
        weights_and_biases_regex: str,
        first_key_ending: str,
        new_key_ending: str,
    ):
        # fuse qkv o weights
        wb_keys = get_matching_keys_for(weights_and_biases_regex, block_keys)
        if len(wb_keys) == 0:
            return state_dict
        # qkv o are sorted here
        tensors_to_fuse = []
        for key in wb_keys:
            tensors_to_fuse.append(state_dict.pop(key))

        # add qkv o weight to state dict
        fused_weight = torch.cat(tensors_to_fuse, dim=0)
        fused_weight_key = wb_keys[0].replace(
            first_key_ending, new_key_ending
        )

        state_dict.update({fused_weight_key: fused_weight})
        return state_dict

    def convert_mlstm_layer_weights_(
        state_dict: dict[str, torch.Tensor], block_keys: list[str]
    ):
        """Modifies the state dict in place."""

        # fuse qkv o weights
        state_dict = concat_weights_and_biases(
            state_dict,
            ".*(q|k|v|ogate_preact).weight",
            ".q.weight",
            ".qkv_opreact.weight",
        )

        # fuse qkv o biases
        state_dict = concat_weights_and_biases(
            state_dict,
            ".*(q|k|v|ogate_preact).bias",
            ".q.bias",
            ".qkv_opreact.bias",
        )

        # fuse if gate weights
        state_dict = concat_weights_and_biases(
            state_dict,
            ".*(i|f)gate_preact.weight",
            ".igate_preact.weight",
            ".ifgate_preact.weight",
        )

        # fuse if gate biases
        state_dict = concat_weights_and_biases(
            state_dict,
            ".*(i|f)gate_preact.bias",
            ".igate_preact.bias",
            ".ifgate_preact.bias",
        )

        return state_dict

    def convert_feedforward_weights(
        state_dict: dict[str, torch.Tensor], block_keys: list[str]
    ):
        """Modifies the state dict in place."""
        # fuse feedforward weights
        state_dict = concat_weights_and_biases(
            state_dict,
            ".*(proj_up_gate|proj_up).weight",
            ".proj_up_gate.weight",
            ".proj_up_gate_z.weight",
        )

        # fuse feedforward biases
        state_dict = concat_weights_and_biases(
            state_dict,
            ".*(proj_up_gate|proj_up).bias",
            ".proj_up_gate.bias",
            ".proj_up_gate_z.bias",
        )

        return state_dict

    # iterate over blocks and convert weights blockwise
    block_idx = 0
    blocks_key_str = "backbone.blocks.{block_idx}"
    state_dict_keys = list(single_weight_state_dict.keys())
    while True:
        block_key_str = blocks_key_str.format(block_idx=block_idx)
        block_keys = get_matching_keys_for(block_key_str, state_dict_keys)
        if len(block_keys) == 0:
            break

        single_weight_state_dict = convert_mlstm_layer_weights_(
            single_weight_state_dict, block_keys
        )

        single_weight_state_dict = convert_feedforward_weights(
            single_weight_state_dict, block_keys
        )

        block_idx += 1

    return single_weight_state_dict
