"""Tools for performance benchmarks"""

from timeit import default_timer


class SequentialTimer:
    def __init__(self):
        self.tt = [("init", default_timer())]

    def add(self, k: str):
        self.tt.append((k, default_timer()))

    def get_diffs(self):
        return [(k, t - tp) for (k, t), (_, tp) in zip(self.tt[1:], self.tt)]

    def __str__(self):
        lines = ["Timings: "] + [
            f"  -{k.ljust(10)}\t {t:.5f} s" for k, t in self.get_diffs()
        ]
        return "\n".join(lines)
