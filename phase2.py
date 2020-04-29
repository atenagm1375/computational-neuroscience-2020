from tests.phase2.phase2 import *

import sys


if __name__ == '__main__':
    try:
        q = sys.argv[1]
        section = sys.argv[2]
    except IndexError:
        print("Enter question number and connection type while running the code.")

    globals()["q{}_{}".format(q, section)]()
