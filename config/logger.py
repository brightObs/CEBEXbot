import logging
import os

def get_logger(name="application"):
    """
    Set up and return a logger instance.
    Logs are written to 'logs/app.log' and also output to the console.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the logs directory exists

    # Create logger
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)  # Adjust the level as needed (DEBUG, INFO, etc.)

    return logger
