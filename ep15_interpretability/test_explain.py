import numpy as np
import credit_explain as c
def test():
    P = c.train_mlp(c.Z[:3000], c.y[:3000], epochs=50); z = c.Zt[0]; base = np.zeros_like(z)
    ig = c.integrated_gradients(P, z, base, steps=128)
    diff = c.mlp_f(P, z[None])[0][0] - c.mlp_f(P, base[None])[0][0]
    assert abs(ig.sum() - diff) < 1e-2   # 完备性：归因之和 = 输出差
    w, _ = c.train_lr(c.Z, c.y); assert w[0] > 0 and w[2] < 0  # 收入正、逾期负
if __name__ == "__main__": test(); print("ep15 tests passed")
