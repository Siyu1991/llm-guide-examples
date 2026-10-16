import random
from multi_agent import single_agent, multi_agent, coder_mock, Meter, run_tests, GOOD, BAD, TASKS
def test():
    for n in TASKS: assert run_tests(GOOD[n], TASKS[n][1]) is None and run_tests(BAD[n], TASKS[n][1])
    s = sum(single_agent(random.Random(1), .3, coder_mock, Meter()) for _ in range(300))
    mm = Meter(); m = sum(multi_agent(random.Random(1), .3, coder_mock, mm) for _ in range(300))
    assert m > s and mm.calls > 300 * 3
if __name__ == "__main__": test(); print("ep11 tests passed")
