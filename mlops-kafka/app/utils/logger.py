import json
import socket
from loguru import logger
import sys

HOSTNAME = socket.gethostname()

def setup_logger(settings):

    def json_sink(message):
        r = message.record

        log = {
            "timestamp": r["time"].timestamp(),
            "level": r["level"].name,
            "service": settings.SERVICE,
            "host": HOSTNAME,
            "event": r["extra"].get("event"),
            "message": r["message"],
            **r["extra"]
        }

        with open(settings.FILE_PATH, "a") as f:
            f.write(json.dumps(log) + "\n")

    logger.remove()

    logger.add(
        json_sink,
        level=settings.LEVEL
    )

    # Console sink (pretty logs)
    logger.add(
        sys.stdout,
        level=settings.LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}"
    )

    return logger