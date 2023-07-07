import os
import re
from inotify_simple import INotify, flags


inotify = INotify()
watch_flags = flags.CREATE | flags.DELETE | flags.MODIFY | flags.DELETE_SELF
wd = inotify.add_watch('../midi/nestup', watch_flags)

# And see the corresponding events:
def watch_nestup():
    while True:
        for event in inotify.read():
            print(event)
            for flag in flags.from_mask(event.mask):
                print('    ' + str(flag))
            generate_midi()

def generate_midi():
    for f in os.listdir("../midi/nestup"):
        if (m := re.fullmatch(r"(.+)\.rhy", f)):
            rhy = f"../midi/nestup/{m.group()}"
            name = m.group(1)
            os.system(f"node nested-tuplets/cli.js -i {rhy} ../midi/{name}")
    os.system("cd .. && make push")

if __name__ == "__main__":
    generate_midi()
    watch_nestup()
