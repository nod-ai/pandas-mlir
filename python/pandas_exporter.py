from enum import Enum, auto

def export_pandas(fn):
    """
    Decorator to define which functions should be exported.
        Args:
            fn (python function): Function to be exported
        Returns:
            Function to be exported
    """
    return fn

def annotate_pandas(*args, **kwargs):
    """
        *args: Original program arguments
        **kwargs: Type and shape information of the arguments
                  and return values.

    Returns:
        Function to be annotated.
    """
    def decorator(fn):
        return fn
    return decorator
