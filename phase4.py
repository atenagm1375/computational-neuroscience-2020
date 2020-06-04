import sys

from tests.phase4.q1 import *


if __name__ == "__main__":
    question_no = sys.argv[1]
    trial_no = sys.argv[2]
    globals()["trial{}".format(trial_no)]()
