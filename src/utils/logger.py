import logging
import sys

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    stream=sys.stdout,
)


def get_logger(name: str, background: bool = False) -> logging.Logger:
    """
    Returns a logger with the given name.
    If background=True, adds a [BACKGROUND] prefix to all messages.
    """
    logger = logging.getLogger(name)
    if background:

        class BackgroundAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                return f"[BACKGROUND] {msg}", kwargs

        return BackgroundAdapter(logger, {})
    return logger
