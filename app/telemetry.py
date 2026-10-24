"""
Prometheus Metrics Exporter and Request Middleware.
"""

import time
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class MetricsCollector:
    """In-memory Prometheus-compatible metrics tracker."""

    def __init__(self):
        self.request_counts: dict[str, int] = defaultdict(int)
        self.request_durations: dict[str, list[float]] = defaultdict(list)
        self.failure_counts: dict[str, int] = defaultdict(int)

    def record_request(self, endpoint: str, duration_sec: float, status_code: int):
        self.request_counts[endpoint] += 1
        self.request_durations[endpoint].append(duration_sec)
        if status_code >= 400:
            self.failure_counts[endpoint] += 1

    def format_prometheus(self) -> str:
        lines = [
            "# HELP ragscalebench_requests_total Total HTTP requests handled.",
            "# TYPE ragscalebench_requests_total counter",
        ]
        for ep, count in self.request_counts.items():
            lines.append(f'ragscalebench_requests_total{{endpoint="{ep}"}} {count}')

        lines.extend(
            [
                "# HELP ragscalebench_failures_total Total failed requests.",
                "# TYPE ragscalebench_failures_total counter",
            ]
        )
        for ep, failures in self.failure_counts.items():
            lines.append(f'ragscalebench_failures_total{{endpoint="{ep}"}} {failures}')

        lines.extend(
            [
                "# HELP ragscalebench_latency_seconds_sum Sum of request durations.",
                "# TYPE ragscalebench_latency_seconds_sum summary",
            ]
        )
        for ep, durations in self.request_durations.items():
            total_sec = sum(durations)
            lines.append(f'ragscalebench_latency_seconds_sum{{endpoint="{ep}"}} {total_sec:.6f}')
            lines.append(f'ragscalebench_latency_seconds_count{{endpoint="{ep}"}} {len(durations)}')

        return "\n".join(lines) + "\n"


global_metrics = MetricsCollector()


class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - t0
        path = request.url.path
        global_metrics.record_request(path, duration, response.status_code)
        response.headers["X-Process-Time"] = f"{duration * 1000.0:.2f}ms"
        return response
