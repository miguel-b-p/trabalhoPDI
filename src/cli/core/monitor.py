"""
System monitoring for training sessions.
"""

import psutil
import GPUtil
import time
import threading
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import logging


class TrainingMonitor:
    """Monitor system resources during training."""
    
    def __init__(self, log_interval: float = 1.0):
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Started system monitoring")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected data."""
        if not self.monitoring:
            return {}
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        stats = self._calculate_stats()
        self._reset_data()
        
        self.logger.info("Stopped system monitoring")
        return stats
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.append({
                    "total": memory.total,
                    "used": memory.used,
                    "available": memory.available,
                    "percent": memory.percent
                })
                
                # GPU usage
                try:
                    gpus = GPUtil.getGPUs()
                    gpu_data = []
                    for gpu in gpus:
                        gpu_data.append({
                            "id": gpu.id,
                            "name": gpu.name,
                            "load": gpu.load * 100,
                            "memory_used": gpu.memoryUsed,
                            "memory_total": gpu.memoryTotal,
                            "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            "temperature": gpu.temperature
                        })
                    self.gpu_usage.append(gpu_data)
                except:
                    self.gpu_usage.append([])
                
                self.timestamps.append(datetime.now())
                
            except Exception as e:
                self.logger.error(f"Error in monitoring: {e}")
            
            time.sleep(self.log_interval)
    
    def _calculate_stats(self) -> Dict[str, Any]:
        """Calculate statistics from collected data."""
        if not self.timestamps:
            return {}
        
        stats = {
            "duration_seconds": (self.timestamps[-1] - self.timestamps[0]).total_seconds(),
            "cpu": self._calculate_cpu_stats(),
            "memory": self._calculate_memory_stats(),
            "gpu": self._calculate_gpu_stats()
        }
        
        return stats
    
    def _calculate_cpu_stats(self) -> Dict[str, float]:
        """Calculate CPU usage statistics."""
        if not self.cpu_usage:
            return {}
        
        return {
            "mean_percent": float(np.mean(self.cpu_usage)),
            "max_percent": float(np.max(self.cpu_usage)),
            "min_percent": float(np.min(self.cpu_usage)),
            "std_percent": float(np.std(self.cpu_usage))
        }
    
    def _calculate_memory_stats(self) -> Dict[str, float]:
        """Calculate memory usage statistics."""
        if not self.memory_usage:
            return {}
        
        percentages = [m["percent"] for m in self.memory_usage]
        
        return {
            "mean_percent": float(np.mean(percentages)),
            "max_percent": float(np.max(percentages)),
            "min_percent": float(np.min(percentages)),
            "peak_usage_gb": float(np.max([m["used"] for m in self.memory_usage]) / (1024**3)),
            "mean_usage_gb": float(np.mean([m["used"] for m in self.memory_usage]) / (1024**3))
        }
    
    def _calculate_gpu_stats(self) -> List[Dict[str, float]]:
        """Calculate GPU usage statistics."""
        if not self.gpu_usage or not self.gpu_usage[0]:
            return []
        
        gpu_stats = []
        num_gpus = len(self.gpu_usage[0])
        
        for gpu_idx in range(num_gpus):
            gpu_data = [g[gpu_idx] for g in self.gpu_usage if len(g) > gpu_idx]
            
            if gpu_data:
                stats = {
                    "id": gpu_data[0]["id"],
                    "name": gpu_data[0]["name"],
                    "mean_load_percent": float(np.mean([g["load"] for g in gpu_data])),
                    "max_load_percent": float(np.max([g["load"] for g in gpu_data])),
                    "mean_memory_percent": float(np.mean([g["memory_percent"] for g in gpu_data])),
                    "max_memory_percent": float(np.max([g["memory_percent"] for g in gpu_data])),
                    "peak_memory_gb": float(np.max([g["memory_used"] for g in gpu_data]) / 1024),
                    "max_temperature": float(np.max([g["temperature"] for g in gpu_data]))
                }
                gpu_stats.append(stats)
        
        return gpu_stats
    
    def _reset_data(self):
        """Reset collected data."""
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.gpu_usage.clear()
        self.timestamps.clear()
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system stats."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            stats = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3)
            }
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    stats["gpu"] = [
                        {
                            "id": gpu.id,
                            "load_percent": gpu.load * 100,
                            "memory_used_gb": gpu.memoryUsed / 1024,
                            "memory_total_gb": gpu.memoryTotal / 1024,
                            "temperature": gpu.temperature
                        }
                        for gpu in gpus
                    ]
            except:
                stats["gpu"] = []
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
