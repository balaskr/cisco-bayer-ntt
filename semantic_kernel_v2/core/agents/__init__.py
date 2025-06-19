import logging

from semantic_kernel.utils.logging import setup_logging

from dotenv import load_dotenv;load_dotenv()

# Configure Semantic Kernel logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)