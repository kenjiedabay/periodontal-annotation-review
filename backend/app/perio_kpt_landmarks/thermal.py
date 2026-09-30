"""NVIDIA GPU telemetry with conservative thermal guards."""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path


class ThermalMonitor:
    def __init__(self, log_path: Path, pause_c: int = 75, resume_c: int = 68,
                 stop_c: int = 80, log_seconds: int = 30, poll_seconds: float = 2.0):
        self.log_path = log_path
        self.pause_c, self.resume_c, self.stop_c = pause_c, resume_c, stop_c
        self.log_seconds, self.poll_seconds = log_seconds, poll_seconds
        self.maximum_temperature = 0
        self.maximum_memory_mib = 0
        self.events: list[dict] = []
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self._finish = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def query() -> tuple[int, int, int, int]:
        command = ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                   "--format=csv,noheader,nounits"]
        output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()[0]
        return tuple(int(value.strip()) for value in output.split(","))  # type: ignore[return-value]

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("timestamp,temp_c,util_percent,memory_used_mib,memory_total_mib\n", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last_log = 0.0
        while not self._finish.is_set():
            try:
                temperature, utilization, used, total = self.query()
                self.maximum_temperature = max(self.maximum_temperature, temperature)
                self.maximum_memory_mib = max(self.maximum_memory_mib, used)
                now = time.time()
                if now - last_log >= self.log_seconds:
                    with self.log_path.open("a", encoding="utf-8") as stream:
                        stream.write(f"{now:.3f},{temperature},{utilization},{used},{total}\n")
                    last_log = now
                if temperature >= self.stop_c and not self.stopped.is_set():
                    self.events.append({"event": "thermal_stop", "temperature_c": temperature, "time": now})
                    self.stopped.set()
                elif temperature >= self.pause_c and not self.paused.is_set():
                    self.events.append({"event": "thermal_pause", "temperature_c": temperature, "time": now})
                    self.paused.set()
                elif self.paused.is_set() and temperature <= self.resume_c:
                    self.events.append({"event": "thermal_resume", "temperature_c": temperature, "time": now})
                    self.paused.clear()
            except Exception as error:
                self.events.append({"event": "telemetry_error", "detail": str(error), "time": time.time()})
            self._finish.wait(self.poll_seconds)

    def wait_if_hot(self) -> bool:
        while self.paused.is_set() and not self.stopped.is_set():
            time.sleep(1)
        return not self.stopped.is_set()

    def close(self) -> None:
        self._finish.set()
        if self._thread:
            self._thread.join(timeout=10)
        try:
            temperature, utilization, used, total = self.query()
            self.maximum_temperature = max(self.maximum_temperature, temperature)
            self.maximum_memory_mib = max(self.maximum_memory_mib, used)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"{time.time():.3f},{temperature},{utilization},{used},{total}\n")
        except Exception:
            pass
