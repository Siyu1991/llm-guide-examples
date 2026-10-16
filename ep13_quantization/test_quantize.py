import numpy as np
from quantize import quant, rel_err, memory_gb
def test():
    w = np.random.default_rng(1).normal(size=(256, 256))
    e = [rel_err(w, quant(w, b)) for b in [8, 4, 2]]; assert e[0] < e[1] < e[2]
    assert rel_err(w, quant(w, 4, group=64)) < rel_err(w, quant(w, 4))
    assert abs(memory_gb(7, 16, 1.0) - 14.0) < 1e-9
if __name__ == "__main__": test(); print("ep13 tests passed")
