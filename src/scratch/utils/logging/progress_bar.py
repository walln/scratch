"""Common progress bar widgets."""

from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn


def get_progress_widgets():
    """Return a list of progress bar widgets."""
    return [
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]
