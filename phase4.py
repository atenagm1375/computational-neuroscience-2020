import sys

from tests.phase4 import q1, q2


if __name__ == "__main__":
    question_no = sys.argv[1]
    trial_no = sys.argv[2]
    fn = getattr(globals()["q{}".format(question_no)], "trial{}".format(trial_no))
    fn()
