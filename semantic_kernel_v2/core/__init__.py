import logging

from semantic_kernel.utils.logging import setup_logging


# Configure Semantic Kernel logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)