from __future__ import annotations

import platform
import sys
from typing import Any, Dict

import psutil


def get_system_snapshot() -> Dict[str, Any]:
    """Collect system metrics without crashing when data cannot be read."""
    try:
        cpu_usage = psutil.cpu_percent(interval=None)
    except Exception:
        cpu_usage = 0.0

    try:
        memory = psutil.virtual_memory()
        ram_usage = round(memory.percent, 1)
        available_ram = round(memory.available / (1024 * 1024 * 1024), 2)
    except Exception:
        ram_usage = 0.0
        available_ram = 0.0

    try:
        disk = psutil.disk_usage('/')
        disk_usage = round(disk.percent, 1)
    except Exception:
        disk_usage = 0.0

    return {
        "cpu_usage": float(cpu_usage),
        "ram_usage": float(ram_usage),
        "disk_usage": float(disk_usage),
        "available_ram": float(available_ram),
        "operating_system": platform.platform(),
        "python_version": sys.version.split()[0],
    }
