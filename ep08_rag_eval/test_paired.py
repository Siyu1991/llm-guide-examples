from paired_and_refusal import mcnemar_exact, kappa
assert mcnemar_exact([1, 1, 0], [1, 1, 0])[2] == 1.0
assert abs(mcnemar_exact([1] * 5 + [0], [0] * 6)[2] - 0.0625) < 1e-9
assert kappa([1, 0, 1, 0], [1, 0, 1, 0])[1] == 1.0
print("ok")
