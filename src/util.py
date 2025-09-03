import os
import sys

import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        pd.to_pickle(obj, file_path)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)