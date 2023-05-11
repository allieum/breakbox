import os
import sys
from collections import defaultdict

class Timer:
    def __init__(self, name):
        self.name = name
        self.time = time.time()

    def tick(self):
        now = time.time()
        print(f"{self.name}: {now - self.time}")
        self.time = now
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret

timer = keydefaultdict(Timer)

def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)
