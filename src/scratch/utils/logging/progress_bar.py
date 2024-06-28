"""Common progress bar widgets."""

from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn


def get_progress_widgets():
    """The default progress bar widgets.

    Returns:
        The default progress bar widgets as a list.
    """
    return [
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]
