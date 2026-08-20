"""
Structured, stage-aware logging shared by every pipeline stage.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def stage(logger: logging.Logger, stage_name: str, **context):
    ctx = " ".join(f"{k}={v}" for k, v in context.items())
    logger.info(f"START  {stage_name} {ctx}".strip())
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception(f"FAILED {stage_name}")
        raise
    else:
        dt = time.perf_counter() - t0
        logger.info(f"DONE   {stage_name} duration_s={dt:.2f}")
