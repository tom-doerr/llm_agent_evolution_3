from typing import TypeVar, Callable, Any
import functools

T = TypeVar('T')

def handle_errors(default_return: T = None, error_msg_prefix: str = "Error") -> Callable:
    """
    Decorator to handle errors consistently.
    
    Args:
        default_return: Value to return if an error occurs
        error_msg_prefix: Prefix for error messages
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except (ValueError, TypeError, RuntimeError) as error:
                # Handle common errors
                print(f"{error_msg_prefix} in {func.__name__}: {error}")
                return default_return
            except Exception as error:
                # Handle unexpected errors
                print(f"Unexpected {error_msg_prefix.lower()} in {func.__name__}: {error}")
                return default_return
        return wrapper
    return decorator

# TODO: Replace try-except blocks with this decorator throughout the codebase
# TODO: Add more specific error handlers for different types of errors
# TODO: Implement logging instead of print statements
