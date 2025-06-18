
import logging

from semantic_kernel.utils.logging import setup_logging

setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)