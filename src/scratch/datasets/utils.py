"""Common utility functions for the datasets module."""

import inspect
import warnings


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
