# src/utils.py
import logging
import numpy as np
import os
import random
import time
import functools

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("WalmartForecast")

def set_seed(seed=42):
    """Thiết lập seed để kết quả ổn định (reproducible)"""
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    except Exception:
        pass
        
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def timer_decorator(func):
    """Decorator để đo thời gian chạy của hàm"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"⏱️ {func.__name__} took {execution_time:.2f}s")
        return result
    return wrapper