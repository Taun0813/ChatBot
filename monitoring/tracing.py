"""
Request tracing and observability for AI Agent
"""
import time
import uuid
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextvars import ContextVar
import json
import threading

logger = logging.getLogger(__name__)

# OpenTelemetry integration (optional)
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Install opentelemetry packages for full tracing support.")

# Context variable for current trace
current_trace: ContextVar[Optional['TraceContext']] = ContextVar('current_trace', default=None)

@dataclass
class Span:
    """A span in a trace"""
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class TraceContext:
    """Trace context for request tracing"""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    ot_span: Optional[Any] = None  # OpenTelemetry span
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def __post_init__(self):
        if not hasattr(self, '_lock'):
            self._lock = threading.RLock()
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None) -> 'SpanContext':
        """Start a new span"""
        span_id = str(uuid.uuid4())
        span = Span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        with self._lock:
            self.spans.append(span)
        
        return SpanContext(self, span)
    
    def add_tag(self, key: str, value: Any) -> None:
        """Add a tag to the trace"""
        with self._lock:
            self.tags[key] = value
    
    def finish(self) -> None:
        """Finish the trace"""
        with self._lock:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary"""
        with self._lock:
            return {
                "trace_id": self.trace_id,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": self.duration,
                "tags": self.tags,
                "spans": [
                    {
                        "span_id": span.span_id,
                        "parent_span_id": span.parent_span_id,
                        "operation_name": span.operation_name,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "duration": span.duration,
                        "tags": span.tags,
                        "logs": span.logs,
                        "error": span.error
                    }
                    for span in self.spans
                ]
            }

class SpanContext:
    """Context manager for spans"""
    
    def __init__(self, trace_context: TraceContext, span: Span):
        self.trace_context = trace_context
        self.span = span
        self._previous_trace = None
    
    def __enter__(self):
        # Store previous trace context
        self._previous_trace = current_trace.get()
        # Set current trace context
        current_trace.set(self.trace_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Finish the span
        self.span.end_time = time.time()
        self.span.duration = self.span.end_time - self.span.start_time
        
        # Add error if exception occurred
        if exc_type is not None:
            self.span.error = str(exc_val)
        
        # Restore previous trace context
        current_trace.set(self._previous_trace)
    
    def add_tag(self, key: str, value: Any) -> None:
        """Add a tag to the span"""
        self.span.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs) -> None:
        """Add a log to the span"""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.span.logs.append(log_entry)
    
    def set_error(self, error: str) -> None:
        """Set error on the span"""
        self.span.error = error

class RequestTracer:
    """Request tracing system"""
    
    def __init__(self, enable_opentelemetry: bool = False, jaeger_endpoint: str = "http://localhost:14268/api/traces"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.active_traces: Dict[str, TraceContext] = {}
        self.completed_traces: List[TraceContext] = []
        self._lock = threading.RLock()
        self.enable_opentelemetry = enable_opentelemetry and OPENTELEMETRY_AVAILABLE
        self.ot_tracer = None
        
        if self.enable_opentelemetry:
            self._setup_opentelemetry(jaeger_endpoint)
    
    def _setup_opentelemetry(self, jaeger_endpoint: str):
        """Setup OpenTelemetry tracing"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": "ai-agent",
                "service.version": "1.0.0"
            })
            
            # Create tracer provider
            trace.set_tracer_provider(TracerProvider(resource=resource))
            
            # Create Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
                collector_endpoint=jaeger_endpoint
            )
            
            # Create span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Get tracer
            self.ot_tracer = trace.get_tracer(__name__)
            self.logger.info("OpenTelemetry tracing initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup OpenTelemetry: {e}")
            self.enable_opentelemetry = False
    
    def start_trace(
        self, 
        operation_name: str = "request",
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> TraceContext:
        """Start a new trace"""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        trace_context = TraceContext(
            trace_id=trace_id,
            start_time=time.time(),
            tags=tags or {}
        )
        
        # Start OpenTelemetry span if enabled
        if self.enable_opentelemetry and self.ot_tracer:
            try:
                ot_span = self.ot_tracer.start_span(operation_name)
                ot_span.set_attributes(tags or {})
                trace_context.ot_span = ot_span
            except Exception as e:
                self.logger.warning(f"Failed to create OpenTelemetry span: {e}")
        
        with self._lock:
            self.active_traces[trace_id] = trace_context
        
        # Set as current trace
        current_trace.set(trace_context)
        
        self.logger.debug(f"Started trace: {trace_id}")
        return trace_context
    
    def finish_trace(self, trace_id: str) -> Optional[TraceContext]:
        """Finish a trace"""
        with self._lock:
            trace_context = self.active_traces.pop(trace_id, None)
            if trace_context:
                trace_context.finish()
                
                # Finish OpenTelemetry span if enabled
                if self.enable_opentelemetry and trace_context.ot_span:
                    try:
                        trace_context.ot_span.end()
                    except Exception as e:
                        self.logger.warning(f"Failed to finish OpenTelemetry span: {e}")
                
                self.completed_traces.append(trace_context)
                
                # Keep only last 1000 traces
                if len(self.completed_traces) > 1000:
                    self.completed_traces = self.completed_traces[-1000:]
                
                self.logger.debug(f"Finished trace: {trace_id}")
                return trace_context
        
        return None
    
    def get_current_trace(self) -> Optional[TraceContext]:
        """Get current trace context"""
        return current_trace.get()
    
    def get_trace(self, trace_id: str) -> Optional[TraceContext]:
        """Get trace by ID"""
        with self._lock:
            return self.active_traces.get(trace_id) or next(
                (t for t in self.completed_traces if t.trace_id == trace_id), 
                None
            )
    
    def get_active_traces(self) -> List[TraceContext]:
        """Get all active traces"""
        with self._lock:
            return list(self.active_traces.values())
    
    def get_completed_traces(self, limit: int = 100) -> List[TraceContext]:
        """Get completed traces"""
        with self._lock:
            return self.completed_traces[-limit:]
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get trace statistics"""
        with self._lock:
            active_count = len(self.active_traces)
            completed_count = len(self.completed_traces)
            
            if completed_count > 0:
                durations = [t.duration for t in self.completed_traces if t.duration is not None]
                avg_duration = sum(durations) / len(durations) if durations else 0
                max_duration = max(durations) if durations else 0
                min_duration = min(durations) if durations else 0
            else:
                avg_duration = max_duration = min_duration = 0
            
            return {
                "active_traces": active_count,
                "completed_traces": completed_count,
                "average_duration": avg_duration,
                "max_duration": max_duration,
                "min_duration": min_duration
            }
    
    def export_traces(self, format: str = "json", limit: int = 100) -> str:
        """Export traces"""
        traces = self.get_completed_traces(limit)
        
        if format == "json":
            return json.dumps([trace.to_dict() for trace in traces], indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_old_traces(self, max_age_seconds: int = 3600) -> int:
        """Cleanup old traces"""
        current_time = time.time()
        cleaned_count = 0
        
        with self._lock:
            # Cleanup completed traces
            traces_to_keep = []
            for trace in self.completed_traces:
                if trace.end_time and (current_time - trace.end_time) < max_age_seconds:
                    traces_to_keep.append(trace)
                else:
                    cleaned_count += 1
            
            self.completed_traces = traces_to_keep
        
        self.logger.info(f"Cleaned up {cleaned_count} old traces")
        return cleaned_count

# Global tracer instance
tracer = RequestTracer(enable_opentelemetry=False)  # Disabled by default

def trace_function(operation_name: Optional[str] = None):
    """Decorator to trace function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get current trace or create new one
            current = current_trace.get()
            if current is None:
                trace = tracer.start_trace(operation_name or func.__name__)
                should_finish = True
            else:
                trace = current
                should_finish = False
            
            # Start span for this function
            with trace.start_span(operation_name or func.__name__) as span:
                try:
                    result = func(*args, **kwargs)
                    span.add_tag("success", True)
                    return result
                except Exception as e:
                    span.set_error(str(e))
                    span.add_tag("success", False)
                    raise
                finally:
                    if should_finish:
                        tracer.finish_trace(trace.trace_id)
        
        return wrapper
    return decorator

def get_current_trace() -> Optional[TraceContext]:
    """Get current trace context"""
    return current_trace.get()

def add_trace_tag(key: str, value: Any) -> None:
    """Add tag to current trace"""
    current = current_trace.get()
    if current:
        current.add_tag(key, value)

def add_trace_log(message: str, level: str = "info", **kwargs) -> None:
    """Add log to current trace"""
    current = current_trace.get()
    if current and current.spans:
        current.spans[-1].add_log(message, level, **kwargs)
