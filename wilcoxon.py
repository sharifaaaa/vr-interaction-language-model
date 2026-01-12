import numpy as np
from scipy.stats import wilcoxon

# -----------------------------------------------------
# 1. Your UCLA pre/post totals (copied from your CSV)
# -----------------------------------------------------

pre  = np.array([
    36,52,38,26,39,30,41,39,35,30,
    42,45,63,51,38,30,29,35,50,35
])

post = np.array([
    37,53,39,26,39,28,43,44,41,30,
    38,51,51,57,33,24,37,31,52,37
])

# -----------------------------------------------------
# 2. Run Wilcoxon signed-rank test
# -----------------------------------------------------

result = wilcoxon(post, pre, zero_method='wilcox', correction=False)

# result.statistic = W
# result.pvalue = p

W = result.statistic
p = result.pvalue

# -----------------------------------------------------
# 3. Compute effect size r = Z / sqrt(N)
# -----------------------------------------------------

# SciPy does not give Z directly; we compute it manually.
# Formula: Z = (W - W_expected) / sqrt(Var(W))

N = len(pre)

# Expected value and variance under H0:
# W_expected = N(N+1) / 4
# Var = N(N+1)(2N+1) / 24

W_expected = N*(N+1)/4
Var_W       = N*(N+1)*(2*N+1)/24
Z           = (W - W_expected) / np.sqrt(Var_W)

# Effect size r
r = Z / np.sqrt(N)

# -----------------------------------------------------
# 4. Print results
# -----------------------------------------------------

print("=== Wilcoxon Signed-Rank Test ===")
print(f"N = {N}")
print(f"W statistic = {W:.4f}")
print(f"Z value = {Z:.4f}")
print(f"p-value = {p:.4f}")
print(f"Effect size (r) = {r:.4f}")

# Direction of change
mean_pre = pre.mean()
mean_post = post.mean()
mean_delta = (post - pre).mean()

print("\n=== Means ===")
print(f"Mean Pre  = {mean_pre:.2f}")
print(f"Mean Post = {mean_post:.2f}")
print(f"Mean Δ    = {mean_delta:.2f}")
