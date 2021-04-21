"""
Intercept log messages from the used libraries and pass them to `loguru`.

See https://github.com/Delgan/loguru

"""
import logging

from loguru import (  # TODO dvp: make loguru optional when mckit is used as library
    logger,
)

# class PropagateHandler(logging.Handler):
#     """Send events from loguru to standard logging"""
#     def emit(self, record):
#         logging.getLogger(record.name).handle(record)
#
#
# logger.add(PropagateHandler(), format="{message}")


class InterceptHandler(logging.Handler):
    """Send events from standard logging to loguru"""

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


log = logging.getLogger()
# log.setLevel(0)
log.addHandler(InterceptHandler())
# logging.basicConfig(handlers=[InterceptHandler()], level=0, style='{')
