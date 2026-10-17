"""OpenTelemetry tracing to the local Phoenix server (`make phoenix`, UI at http://localhost:6006)."""
import os
import urllib.request

from openinference.instrumentation import TraceConfig
from phoenix.otel import register

PHOENIX_URL = os.environ.get("PHOENIX_URL", "http://localhost:6006")
# OpenInference replaces base64 images longer than 32k chars with "__REDACTED__"; a 640x480 frame is ~100k.
MAX_IMAGE_CHARS = 1_000_000


def setup(project: str = "rojev"):
    """Register a global tracer provider exporting to Phoenix. Call `.shutdown()` on the result before exit."""
    try:
        urllib.request.urlopen(PHOENIX_URL, timeout=1)
    except Exception:
        print(f"[tracing] Phoenix not reachable at {PHOENIX_URL}; spans will be dropped (start it with `make phoenix`)")
    return register(project_name=project, endpoint=f"{PHOENIX_URL}/v1/traces", batch=True, verbose=False,
                    config=TraceConfig(base64_image_max_length=MAX_IMAGE_CHARS))
