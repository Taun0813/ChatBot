"""
Metrics collection and monitoring for AI Agent
"""
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MetricData:
    """Metric data structure"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class MetricsConfig:
    """Metrics configuration"""
    enable_metrics: bool = True
    max_metrics_history: int = 1000
    aggregation_window: int = 60  # seconds
    export_interval: int = 300  # seconds
    enable_prometheus: bool = False
    prometheus_port: int = 9090

class MetricsCollector:
    """Collects and manages application metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics: Dict[str, List[MetricData]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        self._start_time = time.time()
    
    def increment_counter(
        self, 
        name: str, 
        value: float = 1.0, 
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric"""
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            self.counters[name] += value
            self._record_metric(MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.COUNTER
            ))
    
    def set_gauge(
        self, 
        name: str, 
        value: float, 
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric"""
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            self.gauges[name] = value
            self._record_metric(MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.GAUGE
            ))
    
    def record_histogram(
        self, 
        name: str, 
        value: float, 
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram metric"""
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            self.histograms[name].append(value)
            self._record_metric(MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM
            ))
    
    def start_timer(self, name: str) -> 'TimerContext':
        """Start a timer metric"""
        return TimerContext(self, name)
    
    def record_timer(
        self, 
        name: str, 
        duration: float, 
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timer metric"""
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            self.timers[name].append(duration)
            self._record_metric(MetricData(
                name=name,
                value=duration,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.TIMER
            ))
    
    def track_api_call(
        self, 
        api_name: str, 
        duration: float, 
        success: bool = True,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Track API call performance"""
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            # Increment counters
            self.increment_counter("total_queries")
            self.increment_counter(f"{api_name}_queries")
            
            if success:
                self.increment_counter("successful_requests")
            else:
                self.increment_counter("failed_requests")
                self.increment_counter(f"{api_name}_errors")
            
            # Record timing
            self.record_timer(f"{api_name}_duration", duration, labels)
            self.record_timer("request_duration", duration, labels)
    
    def track_router_decision(
        self, 
        decision_type: str, 
        duration: float, 
        success: bool = True,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Track router decision performance"""
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            # Increment counters
            self.increment_counter("total_requests")
            self.increment_counter(f"{decision_type}_requests")
            
            if success:
                self.increment_counter("successful_requests")
            else:
                self.increment_counter("failed_requests")
            
            # Record timing
            self.record_timer("request_duration", duration, labels)
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics"""
        if not self.config.enable_metrics:
            return
        
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge("memory_usage_mb", memory.used / (1024 * 1024))
            self.set_gauge("memory_usage_percent", memory.percent)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.set_gauge("cpu_usage_percent", cpu_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.set_gauge("disk_usage_percent", disk_percent)
            
        except ImportError:
            self.logger.warning("psutil not available for system metrics")
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {e}")
    
    def _record_metric(self, metric: MetricData) -> None:
        """Record a metric in history"""
        if len(self.metrics[metric.name]) >= self.config.max_metrics_history:
            self.metrics[metric.name].pop(0)
        
        self.metrics[metric.name].append(metric)
    
    def get_counter(self, name: str) -> float:
        """Get counter value"""
        with self._lock:
            return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> float:
        """Get gauge value"""
        with self._lock:
            return self.gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics"""
        with self._lock:
            values = list(self.histograms.get(name, []))
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "p50": self._percentile(values, 50),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99)
            }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics"""
        with self._lock:
            values = self.timers.get(name, [])
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "p50": self._percentile(values, 50),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99)
            }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: self.get_histogram_stats(name) 
                    for name in self.histograms.keys()
                },
                "timers": {
                    name: self.get_timer_stats(name) 
                    for name in self.timers.keys()
                },
                "uptime": time.time() - self._start_time
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        with self._lock:
            total_requests = self.get_counter("total_requests")
            successful_requests = self.get_counter("successful_requests")
            failed_requests = self.get_counter("failed_requests")
            
            # API-specific metrics
            total_queries = self.get_counter("total_queries")
            rag_queries = self.get_counter("rag_queries")
            conversation_queries = self.get_counter("conversation_queries")
            api_queries = self.get_counter("api_queries")
            
            # Router-specific metrics
            rule_based_requests = self.get_counter("rule_based_requests")
            ml_based_requests = self.get_counter("ml_based_requests")
            hybrid_requests = self.get_counter("hybrid_requests")
            
            success_rate = (
                successful_requests / max(total_requests, 1) * 100
            )
            
            avg_response_time = self.get_timer_stats("request_duration")
            avg_rag_time = self.get_timer_stats("rag_duration")
            avg_conversation_time = self.get_timer_stats("conversation_duration")
            avg_api_time = self.get_timer_stats("api_duration")
            
            # Calculate percentages
            rule_based_percentage = (rule_based_requests / max(total_requests, 1)) * 100
            ml_based_percentage = (ml_based_requests / max(total_requests, 1)) * 100
            hybrid_percentage = (hybrid_requests / max(total_requests, 1)) * 100
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate,
                "average_response_time": avg_response_time.get("mean", 0.0),
                "uptime": time.time() - self._start_time,
                "memory_usage": self.get_gauge("memory_usage_mb"),
                "cpu_usage": self.get_gauge("cpu_usage_percent"),
                # Query breakdown
                "total_queries": total_queries,
                "rag_queries": rag_queries,
                "conversation_queries": conversation_queries,
                "api_queries": api_queries,
                # Router performance
                "rule_based_requests": rule_based_requests,
                "ml_based_requests": ml_based_requests,
                "hybrid_requests": hybrid_requests,
                "rule_based_percentage": rule_based_percentage,
                "ml_based_percentage": ml_based_percentage,
                "hybrid_percentage": hybrid_percentage,
                # Performance metrics
                "avg_rag_time": avg_rag_time.get("mean", 0.0),
                "avg_conversation_time": avg_conversation_time.get("mean", 0.0),
                "avg_api_time": avg_api_time.get("mean", 0.0),
                # Error rates
                "error_rate": (failed_requests / max(total_requests, 1)) * 100,
                "rag_error_rate": (self.get_counter("rag_errors") / max(rag_queries, 1)) * 100,
                "conversation_error_rate": (self.get_counter("conversation_errors") / max(conversation_queries, 1)) * 100,
                "api_error_rate": (self.get_counter("api_errors") / max(api_queries, 1)) * 100
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            self.metrics.clear()
            self._start_time = time.time()
            self.logger.info("Metrics reset")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format == "json":
            return json.dumps(self.get_all_metrics(), indent=2)
        elif format == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        with self._lock:
            # Export counters
            for name, value in self.counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            
            # Export gauges
            for name, value in self.gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
            
            # Export histograms
            for name, stats in self.histograms.items():
                if stats:
                    lines.append(f"# TYPE {name} histogram")
                    lines.append(f"{name}_count {len(stats)}")
                    lines.append(f"{name}_sum {sum(stats)}")
                    lines.append(f"{name}_min {min(stats)}")
                    lines.append(f"{name}_max {max(stats)}")
                    lines.append(f"{name}_mean {sum(stats) / len(stats)}")
        
        return "\n".join(lines)

class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str):
        self.collector = collector
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration)
