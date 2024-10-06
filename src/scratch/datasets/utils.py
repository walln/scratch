"""Common utility functions for the datasets module."""

import inspect
import warnings
from dataclasses import dataclass, field

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def patch_datasets_warning():
    """Patch the warning message for datasets.

    The warning message is due to a bug? in the huggingface datasets library.
    It might be logically correct but pytorch warns on using the spread operator to copy
    tensors rather than using the clone method.
    """
    warning_message = "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)."  # noqa: E501
    warnings.filterwarnings("ignore", message=warning_message, category=UserWarning)

    def filter_specific_warning(warning):
        """Custom filter function to ignore a warning coming from a specific file."""
        if warning.category is UserWarning:
            # Get the current frame
            frame = inspect.currentframe()
            # Go back to the frame where the warning was issued
            while frame:
                filename = frame.f_code.co_filename
                if filename.endswith("datasets/formatting/torch_formatter.py"):
                    print("HERE")

                    return True
                frame = frame.f_back
        return False

    # Register the custom filter
    warnings.filterwarnings("ignore", category=UserWarning, module=r".*")
    warnings.showwarning = (
        lambda message, category, filename, lineno, file=None, line=None: None
        if filter_specific_warning(
            warnings.WarningMessage(message, category, filename, lineno)
        )
        else warnings.showwarning(message, category, filename, lineno)
    )


@dataclass
class TokenizerMetadata:
    """Metadata for a tokenizer."""

    vocab_size: int
    max_length: int

    pad_token_id: int | None = field(default=None)
    bos_token_id: int | None = field(default=None)
    eos_token_id: int | None = field(default=None)
    unk_token_id: int | None = field(default=None)
    cls_token_id: int | None = field(default=None)
    sep_token_id: int | None = field(default=None)
    mask_token_id: int | None = field(default=None)

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase, max_length: int):
        """Create metadata from a tokenizer instance."""
        vocab_size = tokenizer.vocab_size  # type: ignore
        if not vocab_size:
            raise ValueError("The tokenizer does not have a vocab size.")
        return cls(
            vocab_size=vocab_size,
            pad_token_id=cls._safe_get_token_id(tokenizer, "pad_token_id"),
            bos_token_id=cls._safe_get_token_id(tokenizer, "bos_token_id"),
            eos_token_id=cls._safe_get_token_id(tokenizer, "eos_token_id"),
            unk_token_id=cls._safe_get_token_id(tokenizer, "unk_token_id"),
            cls_token_id=cls._safe_get_token_id(tokenizer, "cls_token_id"),
            sep_token_id=cls._safe_get_token_id(tokenizer, "sep_token_id"),
            mask_token_id=cls._safe_get_token_id(tokenizer, "mask_token_id"),
            max_length=max_length,
        )

    @staticmethod
    def _safe_get_token_id(
        tokenizer: PreTrainedTokenizerBase, token_attr: str
    ) -> int | None:
        """Safely get a token id from the tokenizer, return None if not available."""
        return getattr(tokenizer, token_attr, None)


def load_tokenizer(name_or_path: str, max_length: int = 512):
    """Load a tokenizer from the Hugging Face model hub.

    Args:
        name_or_path: the name or path of the tokenizer
        max_length: the maximum length of the sequences

    Returns:
        The tokenizer object
    """
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path, clean_up_tokenization_spaces=True
    )
    tokenizer.model_max_length = max_length

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer


def get_tokenizer_metadata(
    tokenizer: PreTrainedTokenizerBase, max_length: int
) -> TokenizerMetadata:
    """Get the metadata for a tokenizer.

    Args:
        tokenizer: the tokenizer object
        max_length: the maximum length of the sequences
    """
    return TokenizerMetadata.from_tokenizer(tokenizer, max_length)
