import logging

logger = logging.getLogger()
log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger.handlers = [console_handler]