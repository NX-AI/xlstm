from typing import Callable

import torch
from torch.profiler import record_function

LLMStateType = dict

LLMForwardFnType = Callable[
    [torch.Tensor, LLMStateType | None], tuple[torch.Tensor, LLMStateType]
]


def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """Takes the argmax of the logits. Actually no sampling is done here."""
    return torch.argmax(logits, dim=-1)

# Ths registry is used to map the sampling function name to the actual function
_sampling_fn_registry = {
    "greedy": greedy_sampling,
}


def get_sampling_fn(sampling_fn_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return _sampling_fn_registry[sampling_fn_name]


@torch.no_grad
def generate_tokens(
    llm_forward: LLMForwardFnType,
    prefill_tokens: torch.Tensor | None = None,
    max_length: int = 128,
    token_sample_fn: Callable[[torch.Tensor], torch.Tensor] = greedy_sampling,
    bos_token_id: int = 0,
    state: LLMStateType | None = None,
    batch_size_no_prefill: int = 1,
    generated_tokens: torch.Tensor | None = None,
    device: str = "cuda",
) -> tuple[torch.Tensor, LLMStateType]:
    """A simple function to generate tokens from a language model.
    It generates exactly max_length tokens.

    Args:
        llm_forward: The forward function of the language model.
        prefill_tokens: The tokens to prefill the model with.
        max_length: The maximum length of the generated tokens.
        token_sample_fn: The function to sample the next token.
        bos_token_id: The id of the beginning of sequence token.
        state: The state of the language model.
        batch_size_no_prefill: The batch size used for generation if no prefill_tokens are given.
        generated_tokens: The generated tokens tensor.
        device: The device to use.

    Returns:
        generated_tokens: The generated tokens tensor.
        state: The state of the language model after generation.
    """

    if prefill_tokens is None:
        prefill_tokens = torch.full((batch_size_no_prefill, 1), fill_value=bos_token_id, dtype=torch.long, device=device)
    
    if prefill_tokens.ndim == 1:
        prefill_tokens = prefill_tokens[:, None]

    batch_size = prefill_tokens.size(0)

    if max_length > 0:
        # init the generated tokens tensors
        # we add 1 to the max_length to account for the BOS token
        if generated_tokens is None:
            generated_tokens = torch.empty(
                (batch_size, max_length), dtype=torch.long, device=device
            )
        else:
            assert generated_tokens.shape == (batch_size, max_length), (
                f"Generated tokens shape: {tuple(generated_tokens.shape)}, "
                f"expected {(batch_size, max_length)}"
            )

        last_token = prefill_tokens
        for i in range(max_length):
            with record_function(f"generate_tokens_step_{i}"):
                logits, state = llm_forward(last_token, state)
                next_token = token_sample_fn(logits[:, -1:])
                generated_tokens[:, i:i+1] = next_token
                last_token = next_token
    else:
        # we only return the state
        generated_tokens = None

    return generated_tokens, state