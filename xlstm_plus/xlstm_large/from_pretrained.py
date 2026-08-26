from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import load_file

from .model import xLSTMLarge, xLSTMLargeConfig


def load_from_pretrained(
    checkpoint_path: str | Path,
    return_last_states: bool | None = None,
    chunkwise_kernel_name: str | None = None,
    sequence_kernel_name: str | None = None,
    step_kernel_name: str | None = None,
    backend_mode: str | None = "inference",
    chunk_size: int | None = None,
) -> xLSTMLarge:
    """
    Load a mLSTM model from a checkpoint.

    Args:
        checkpoint_path: The path to the checkpoint.
        return_last_states: Whether to return the last states. Overwrites the value from loaded config.
        chunkwise_kernel_name: The chunkwise kernel to use. Overwrites the value from loaded config.
        sequence_kernel_name: The sequence kernel to use. Overwrites the value from loaded config.
        step_kernel_name: The step kernel to use. Overwrites the value from loaded config.
        backend_mode: The backend mode to use. Overwrites the value from loaded config.
        chunk_size: The chunk size to use. Overwrites the value from loaded config.

    Returns:
        mLSTM: The loaded mLSTM model.
    """

    checkpoint_path = Path(checkpoint_path)
    non_sharded_path = checkpoint_path / "model.safetensors"
    if non_sharded_path.exists():
        state_dict = load_file(non_sharded_path)
    else:
        n = 0
        sharded_path = checkpoint_path / f"model_{n}.safetensors"
        state_dict = {}
        while sharded_path.exists():
            state_dict.update(load_file(sharded_path))
            n += 1
            sharded_path = checkpoint_path / f"model_{n}.safetensors"
    config = OmegaConf.load(checkpoint_path / "config.yaml")

    mlstm_config = xLSTMLargeConfig(**config)
    # Note: The default weight mode is single.
    # For fused weight mode convert the weights using convert_single_weights_to_fused_weights.
    mlstm_config.weight_mode = "single"
    if chunkwise_kernel_name is not None:
        mlstm_config.chunkwise_kernel = chunkwise_kernel_name
    if sequence_kernel_name is not None:
        mlstm_config.sequence_kernel = sequence_kernel_name
    if step_kernel_name is not None:
        mlstm_config.step_kernel = step_kernel_name
    if backend_mode is not None:
        mlstm_config.mode = backend_mode
    if return_last_states is not None:
        mlstm_config.return_last_states = return_last_states
    if chunk_size is not None:
        mlstm_config.chunk_size = chunk_size

    mlstm = xLSTMLarge(mlstm_config)
    mlstm.load_state_dict(state_dict)

    return mlstm