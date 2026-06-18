import os
import time
import psutil
from pathlib import Path
from contextlib import ContextDecorator
try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None
BASE_DIR = Path.cwd()
ARTIFACTS_DIR = BASE_DIR / "artifacts"
class ProfileResource(ContextDecorator):
    """
    A unified profiler that can be used either as a decorator or a context manager.
    Tracks execution latency, carbon emissions, CPU percentage, and RAM usage.
    ```
    {
        "latency_ms"
        "emissions_kg_co2"
        "ram_delta_mb"
        "ram_total_mb"
        "cpu_utilization_pct"
        "work_dir"
    }
    ```
    """
    def __init__(self, project_name: str, method_name: str, artifacts_dir: Path=ARTIFACTS_DIR,enabled: bool = True):
        self.project_name = project_name
        self.method_name = method_name
        self.work_dir = Path(artifacts_dir) / project_name / method_name
        self.enabled = enabled
        self.tracker = None
        self.process = psutil.Process(os.getpid())
        self.metrics = {}
    def __enter__(self):
        if not self.enabled:
            return self
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # 1. Capture system state before run
        self.mem_before = (
            self.process.memory_info().rss
        )  # Resident Set Size (Physical memory allocated)
        self.process.cpu_percent(interval=None)  # Initialize CPU utilization counters
        # 2. Start Carbon tracking
        if EmissionsTracker is not None:
            self.tracker = EmissionsTracker(
                project_name=f"{self.project_name}_{self.method_name}",
                output_dir=str(self.work_dir),
                log_level="error",
            )
            self.tracker.start()
        self.start_time = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            self.metrics = {
                "latency_ms": 0.0,
                "emissions_kg_co2": 0.0,
                "ram_delta_mb": 0.0,
                "ram_total_mb": 0.0,
                "cpu_utilization_pct": 0.0,
                "work_dir": self.work_dir
            }
            return False # Let any internal exceptions bubble up normally
        # 1. Stop core timers and emissions tracking immediately
        duration_sec = time.perf_counter() - self.start_time
        emissions_kg_co2 = 0.0
        if self.tracker:
            try:
                emissions_kg_co2 = self.tracker.stop() or 0.0
            except Exception:
                pass
        # 2. Capture system state after run
        mem_after = self.process.memory_info().rss
        cpu_usage = self.process.cpu_percent(interval=None)
        # 3. Calculate Deltas
        # RSS tells us how much physical RAM has been grabbed or cleared by the process
        mem_delta_mb = (mem_after - self.mem_before) / (1024 * 1024)
        total_mem_mb = mem_after / (1024 * 1024)
        # 4. Save metrics payload internally
        self.metrics = {
            "latency_ms": round(duration_sec * 1000.0, 3),
            "emissions_kg_co2": round(emissions_kg_co2, 8),
            "ram_delta_mb": round(mem_delta_mb, 4),
            "ram_total_mb": round(total_mem_mb, 4),
            "cpu_utilization_pct": round(cpu_usage, 2),
            "work_dir": self.work_dir,
        }
        return False  # Ensure exceptions scale upward properly
