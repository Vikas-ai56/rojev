"""One thread per simulated robot that owns all of its OpenGL contexts."""
import queue
import threading


class GLThread:
    """The one thread that owns a robot's OpenGL contexts, like a camera driver owns its device. Renderers are created,
    rendered and closed only through `call`: a macOS CGL context used from a second thread blocks forever, so readers on
    any thread (control loop, evaluators) hand their GL work to this thread. `update_scene` is CPU-only and stays with
    the caller, under the physics lock."""

    def __init__(self, name: str):
        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def call(self, fn, *args):
        if threading.current_thread() is self._thread:
            return fn(*args)
        box: dict = {}
        done = threading.Event()
        self._jobs.put((fn, args, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["value"]

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join()

    def _run(self) -> None:
        while (job := self._jobs.get()) is not None:
            fn, args, box, done = job
            try:
                box["value"] = fn(*args)
            except BaseException as e:
                box["error"] = e
            finally:
                done.set()
