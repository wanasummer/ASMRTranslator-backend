import io
import threading
import time
import unittest

from modules.pipeline_logger import PipelineTee


class _BrokenOutput:
    def write(self, _data):
        raise OSError(22, "Invalid argument")

    def flush(self):
        raise OSError(22, "Invalid argument")


class _FlushBrokenOutput:
    def __init__(self):
        self.data = ""

    def write(self, data):
        self.data += data

    def flush(self):
        raise OSError(22, "Invalid argument")


class _OverlapDetectingOutput:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._state_lock = threading.Lock()

    def write(self, _data):
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.002)
        with self._state_lock:
            self.active -= 1

    def flush(self):
        pass


class PipelineTeeTests(unittest.TestCase):
    def test_broken_original_write_does_not_interrupt_log(self):
        log = io.StringIO()
        tee = PipelineTee(_BrokenOutput(), log)

        self.assertEqual(tee.write("pipeline output\n"), 16)
        tee.flush()

        self.assertEqual(log.getvalue(), "pipeline output\n")

    def test_broken_original_flush_does_not_interrupt_log(self):
        original = _FlushBrokenOutput()
        log = io.StringIO()
        tee = PipelineTee(original, log)

        tee.write("completed\n")
        tee.flush()
        tee.write("still running\n")

        self.assertEqual(original.data, "completed\n")
        self.assertEqual(log.getvalue(), "completed\nstill running\n")

    def test_concurrent_writes_are_serialized(self):
        original = _OverlapDetectingOutput()
        log = io.StringIO()
        tee = PipelineTee(original, log)
        barrier = threading.Barrier(8)

        def write_from_worker(index):
            barrier.wait()
            tee.write(f"线程-{index}\n")

        threads = [threading.Thread(target=write_from_worker, args=(index,))
                   for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(original.max_active, 1)
        self.assertEqual(len(log.getvalue().splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
