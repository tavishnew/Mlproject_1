import logging
import os
import datetime as dt

log_file = f"{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", log_file)
os.makedirs(os.path.dirname(logs_path), exist_ok=True)
log_file_path = os.path.join(logs_path)

logging.basicConfig(
    filename=log_file_path, 
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO
)