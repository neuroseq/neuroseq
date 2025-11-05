import logging
import os

import psutil
import pynvml
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QLabel


class ResourceMonitorLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.process.cpu_percent(interval=None)
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError as e:
            self.logger.warning(f"Could not initialize NVML: {e}")
            self.handle = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)
        self.update_stats()

    def update_stats(self):
        try:
            cpu_percent = self.process.cpu_percent(interval=None)
            ram_gb = self.process.memory_info().rss / (1024 * 1024 * 1024)
            text = f"CPU: {cpu_percent:.1f}% | RAM: {ram_gb:.2f} GB "
            if self.handle is not None:
                gpu_percent = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                vram_gb = 0
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                    if p.pid == self.pid:
                        vram_gb = p.usedGpuMemory / (1024 * 1024 * 1024)
                text += f"| GPU: {gpu_percent:.1f}% | VRAM: {vram_gb:.2f} GB "
            self.setText(text)
        except Exception as e:
            self.logger.exception(e)
