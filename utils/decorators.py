import time
import logging
from functools import wraps
from typing import Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: tuple = (Exception,)):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                                 f"Retrying in {current_delay:.1f}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        
        return wrapper
    return decorator


def timing(func: Callable) -> Callable:
    """
    Measure and log function execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        logger.debug(f"{func.__name__} executed in {elapsed_time:.4f}s")
        
        return result
    
    return wrapper


def log_execution(level: str = 'INFO'):
    """
    Log function entry and exit
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            log_func = getattr(logger, level.lower())
            
            # Log entry
            log_func(f"→ Entering {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                log_func(f"← Exiting {func.__name__} successfully")
                return result
            
            except Exception as e:
                logger.error(f"✗ {func.__name__} failed with error: {e}")
                raise
        
        return wrapper
    return decorator


def validate_args(**validators):
    """
    Validate function arguments
    
    Example:
        @validate_args(symbol=lambda x: isinstance(x, str) and len(x) > 0,
                      lot_size=lambda x: isinstance(x, float) and x > 0)
        def place_order(symbol, lot_size):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument
            for arg_name, validator in validators.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    
                    if not validator(value):
                        raise ValueError(f"Validation failed for argument '{arg_name}' with value: {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def cache_result(ttl: int = 60):
    """
    Cache function result for specified time (in seconds)
    
    Args:
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from args and kwargs
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            # Check if cached and not expired
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl:
                    logger.debug(f"Returning cached result for {func.__name__}")
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            
            # Clean old cache entries
            cache_keys = list(cache.keys())
            for k in cache_keys:
                _, timestamp = cache[k]
                if current_time - timestamp >= ttl:
                    del cache[k]
            
            return result
        
        return wrapper
    return decorator


def rate_limit(calls: int = 10, period: int = 60):
    """
    Rate limit function calls
    
    Args:
        calls: Maximum number of calls
        period: Time period in seconds
    """
    def decorator(func: Callable) -> Callable:
        call_times = []
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_time = time.time()
            
            # Remove old calls outside the time window
            call_times[:] = [t for t in call_times if current_time - t < period]
            
            # Check if rate limit exceeded
            if len(call_times) >= calls:
                sleep_time = period - (current_time - call_times[0])
                logger.warning(f"Rate limit reached for {func.__name__}. Sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                call_times.pop(0)
            
            # Record this call
            call_times.append(current_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def handle_exceptions(default_return: Any = None, log_traceback: bool = True):
    """
    Catch exceptions and return default value
    
    Args:
        default_return: Value to return on exception
        log_traceback: Whether to log full traceback
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                if log_traceback:
                    logger.exception(f"Exception in {func.__name__}: {e}")
                else:
                    logger.error(f"Exception in {func.__name__}: {e}")
                
                return default_return
        
        return wrapper
    return decorator


def deprecated(reason: str = ""):
    """
    Mark function as deprecated
    
    Args:
        reason: Reason for deprecation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            warning_msg = f"{func.__name__} is deprecated"
            if reason:
                warning_msg += f": {reason}"
            
            logger.warning(warning_msg)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def singleton(cls):
    """
    Singleton class decorator
    """
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


if __name__ == "__main__":
    # Test decorators
    logging.basicConfig(level=logging.DEBUG)
    
    # Test retry decorator
    @retry(max_attempts=3, delay=0.5)
    def unstable_function():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success"
    
    # Test timing decorator
    @timing
    def slow_function():
        time.sleep(0.1)
        return "Done"
    
    # Test cache decorator
    @cache_result(ttl=5)
    def expensive_calculation(x):
        time.sleep(0.5)
        return x ** 2
    
    print("Testing retry...")
    try:
        result = unstable_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    print("\nTesting timing...")
    slow_function()
    
    print("\nTesting cache...")
    print(expensive_calculation(5))  # Slow
    print(expensive_calculation(5))  # Fast (cached)
    
    # Test config loader
    config = ConfigLoader()
    print(f"\nDefault symbol: {config.get('trading.default_symbol')}")
    print(f"Max risk: {config.get('risk_management.default_risk_percent')}%")