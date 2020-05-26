import sys


if __name__ == "__main__":
    question_no = sys.argv[1]
    exec(open("tests/phase4/q{}.py".format(question_no)).read())
