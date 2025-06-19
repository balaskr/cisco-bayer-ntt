import logging

from semantic_kernel.utils.logging import setup_logging

from .sites_tasks_plugin import SiteTasksPlugin
from .delagation_plugin import DelegationPlugin

# Configure Semantic Kernel logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)