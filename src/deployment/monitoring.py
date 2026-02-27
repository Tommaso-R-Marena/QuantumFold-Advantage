import time

from prometheus_client import Counter, Gauge, Histogram

prediction_counter = Counter("predictions_total", "Total predictions")
prediction_duration = Histogram("prediction_duration_seconds", "Prediction time")
gpu_utilization = Gauge("gpu_utilization_percent", "GPU usage")
queue_length = Gauge("queue_length", "Number of pending jobs")


def record_prediction(duration_s: float) -> None:
    prediction_counter.inc()
    prediction_duration.observe(duration_s)


def timed_prediction(fn, *args, **kwargs):
    start = time.time()
    out = fn(*args, **kwargs)
    record_prediction(time.time() - start)
    return out
