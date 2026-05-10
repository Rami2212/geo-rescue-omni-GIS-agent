"""Centralised logging configuration for the GeoRescue application."""

import logging
import os


def setup_logging(name: str = "georescue") -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress urllib3 retry noise — agents.tools already logs the final
    # "API unreachable" warning, so these intermediate retry lines add no value.
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    return logging.getLogger(name)
