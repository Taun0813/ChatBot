"""
Health check system for AI Agent
"""
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import psutil
import json

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    response_time: float

@dataclass
class HealthCheckConfig:
    """Health check configuration"""
    enable_system_checks: bool = True
    enable_custom_checks: bool = True
    check_interval: int = 30  # seconds
    timeout: int = 10  # seconds
    memory_threshold: float = 90.0  # percent
    cpu_threshold: float = 90.0  # percent
    disk_threshold: float = 90.0  # percent

class HealthChecker:
    """Health check system for monitoring system status"""
    
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.custom_checks: Dict[str, Callable] = {}
        self.last_check_time = 0
        self.cached_results: Dict[str, HealthCheckResult] = {}
        self.is_running = False
    
    def register_check(self, name: str, check_func: Callable, *args, **kwargs) -> None:
        """Register a custom health check"""
        # Create a wrapper function that passes the arguments
        def wrapped_check():
            return check_func(*args, **kwargs)
        self.custom_checks[name] = wrapped_check
        self.logger.info(f"Registered health check: {name}")
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        results = {}
        
        # Run system checks
        if self.config.enable_system_checks:
            system_results = await self._run_system_checks()
            results.update(system_results)
        
        # Run custom checks
        if self.config.enable_custom_checks:
            custom_results = await self._run_custom_checks()
            results.update(custom_results)
        
        # Cache results
        self.cached_results = results
        self.last_check_time = time.time()
        
        return results
    
    async def _run_system_checks(self) -> Dict[str, HealthCheckResult]:
        """Run system-level health checks"""
        results = {}
        
        # Memory check
        try:
            memory_result = await self._check_memory()
            results["memory"] = memory_result
        except Exception as e:
            results["memory"] = HealthCheckResult(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {e}",
                details={},
                timestamp=time.time(),
                response_time=0.0
            )
        
        # CPU check
        try:
            cpu_result = await self._check_cpu()
            results["cpu"] = cpu_result
        except Exception as e:
            results["cpu"] = HealthCheckResult(
                name="cpu",
                status=HealthStatus.UNHEALTHY,
                message=f"CPU check failed: {e}",
                details={},
                timestamp=time.time(),
                response_time=0.0
            )
        
        # Disk check
        try:
            disk_result = await self._check_disk()
            results["disk"] = disk_result
        except Exception as e:
            results["disk"] = HealthCheckResult(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {e}",
                details={},
                timestamp=time.time(),
                response_time=0.0
            )
        
        return results
    
    async def _run_custom_checks(self) -> Dict[str, HealthCheckResult]:
        """Run custom health checks"""
        results = {}
        
        for name, check_func in self.custom_checks.items():
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    check_func(),
                    timeout=self.config.timeout
                )
                response_time = time.time() - start_time
                
                if isinstance(result, HealthCheckResult):
                    results[name] = result
                else:
                    # Convert simple result to HealthCheckResult
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    results[name] = HealthCheckResult(
                        name=name,
                        status=status,
                        message="Custom check completed",
                        details={},
                        timestamp=time.time(),
                        response_time=response_time
                    )
                
            except asyncio.TimeoutError:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check timed out after {self.config.timeout}s",
                    details={},
                    timestamp=time.time(),
                    response_time=self.config.timeout
                )
            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                    details={},
                    timestamp=time.time(),
                    response_time=0.0
                )
        
        return results
    
    async def check_application_health(self, router_instance=None) -> HealthCheckResult:
        """Check application-specific health"""
        start_time = time.time()
        
        try:
            # Check if router is initialized
            if router_instance is None:
                return HealthCheckResult(
                    name="application",
                    status=HealthStatus.UNHEALTHY,
                    message="Router not initialized",
                    details={"router_status": "not_initialized"},
                    timestamp=time.time(),
                    response_time=time.time() - start_time
                )
            
            # Check router health
            router_healthy = await router_instance.health_check()
            
            if router_healthy:
                status = HealthStatus.HEALTHY
                message = "Application is healthy"
            else:
                status = HealthStatus.DEGRADED
                message = "Application is degraded"
            
            return HealthCheckResult(
                name="application",
                status=status,
                message=message,
                details={
                    "router_status": "initialized",
                    "router_healthy": router_healthy
                },
                timestamp=time.time(),
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="application",
                status=HealthStatus.UNHEALTHY,
                message=f"Application check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                response_time=time.time() - start_time
            )
    
    async def _check_memory(self) -> HealthCheckResult:
        """Check memory usage"""
        start_time = time.time()
        
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > self.config.memory_threshold:
            status = HealthStatus.UNHEALTHY
            message = f"Memory usage is high: {memory_percent:.1f}%"
        elif memory_percent > self.config.memory_threshold * 0.8:
            status = HealthStatus.DEGRADED
            message = f"Memory usage is elevated: {memory_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage is normal: {memory_percent:.1f}%"
        
        return HealthCheckResult(
            name="memory",
            status=status,
            message=message,
            details={
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "used_memory_gb": memory.used / (1024**3),
                "memory_percent": memory_percent
            },
            timestamp=time.time(),
            response_time=time.time() - start_time
        )
    
    async def _check_cpu(self) -> HealthCheckResult:
        """Check CPU usage"""
        start_time = time.time()
        
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > self.config.cpu_threshold:
            status = HealthStatus.UNHEALTHY
            message = f"CPU usage is high: {cpu_percent:.1f}%"
        elif cpu_percent > self.config.cpu_threshold * 0.8:
            status = HealthStatus.DEGRADED
            message = f"CPU usage is elevated: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage is normal: {cpu_percent:.1f}%"
        
        return HealthCheckResult(
            name="cpu",
            status=status,
            message=message,
            details={
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            timestamp=time.time(),
            response_time=time.time() - start_time
        )
    
    async def _check_disk(self) -> HealthCheckResult:
        """Check disk usage"""
        start_time = time.time()
        
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        if disk_percent > self.config.disk_threshold:
            status = HealthStatus.UNHEALTHY
            message = f"Disk usage is high: {disk_percent:.1f}%"
        elif disk_percent > self.config.disk_threshold * 0.8:
            status = HealthStatus.DEGRADED
            message = f"Disk usage is elevated: {disk_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage is normal: {disk_percent:.1f}%"
        
        return HealthCheckResult(
            name="disk",
            status=status,
            message=message,
            details={
                "total_disk_gb": disk.total / (1024**3),
                "used_disk_gb": disk.used / (1024**3),
                "free_disk_gb": disk.free / (1024**3),
                "disk_percent": disk_percent
            },
            timestamp=time.time(),
            response_time=time.time() - start_time
        )
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.cached_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.cached_results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        overall_status = self.get_overall_status()
        
        # Calculate health score
        total_checks = len(self.cached_results)
        healthy_checks = len([r for r in self.cached_results.values() if r.status == HealthStatus.HEALTHY])
        degraded_checks = len([r for r in self.cached_results.values() if r.status == HealthStatus.DEGRADED])
        unhealthy_checks = len([r for r in self.cached_results.values() if r.status == HealthStatus.UNHEALTHY])
        
        health_score = 0
        if total_checks > 0:
            health_score = ((healthy_checks * 100) + (degraded_checks * 50)) / total_checks
        
        return {
            "overall_status": overall_status.value,
            "health_score": health_score,
            "last_check_time": self.last_check_time,
            "check_age_seconds": time.time() - self.last_check_time,
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "degraded_checks": degraded_checks,
            "unhealthy_checks": unhealthy_checks,
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time": result.response_time,
                    "timestamp": result.timestamp
                }
                for name, result in self.cached_results.items()
            }
        }
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information"""
        return {
            "summary": self.get_health_summary(),
            "detailed_results": {
                name: {
                    "name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp,
                    "response_time": result.response_time
                }
                for name, result in self.cached_results.items()
            }
        }
    
    async def start_periodic_checks(self) -> None:
        """Start periodic health checks"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting periodic health checks")
        
        while self.is_running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.config.check_interval)
            except Exception as e:
                self.logger.error(f"Error in periodic health checks: {e}")
                await asyncio.sleep(self.config.check_interval)
    
    def stop_periodic_checks(self) -> None:
        """Stop periodic health checks"""
        self.is_running = False
        self.logger.info("Stopped periodic health checks")
    
    def export_health(self, format: str = "json") -> str:
        """Export health information"""
        if format == "json":
            return json.dumps(self.get_detailed_health(), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
