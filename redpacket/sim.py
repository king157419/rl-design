"""二倍均值法拼手气红包仿真：每个位置的期望、标准差、成为手气最佳的概率。"""
import numpy as np

def simulate(total=100.0, n=10, trials=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    cents = np.full(trials, int(round(total * 100)))
    amounts = np.zeros((trials, n))
    for i in range(n - 1):
        left = n - i
        upper = np.maximum(2 * cents // left, 1)  # 二倍均值上限（分）
        draw = rng.integers(1, upper + 1)          # [1, upper] 均匀取整分
        draw = np.minimum(draw, cents - (left - 1))  # 保证后面每人至少 1 分
        amounts[:, i] = draw
        cents -= draw
    amounts[:, -1] = cents
    amounts /= 100
    best = amounts.argmax(axis=1)
    p_best = np.bincount(best, minlength=n) / trials
    return amounts.mean(0), amounts.std(0), p_best

if __name__ == "__main__":
    for n in (5, 10):
        mean, std, p = simulate(n=n)
        print(f"\n100 元 {n} 个红包，1e6 次仿真；均分概率 1/{n} = {1/n:.3f}")
        print("位置  期望(元)  标准差(元)  手气最佳概率")
        for k in range(n):
            print(f"{k+1:>4}  {mean[k]:8.2f}  {std[k]:9.2f}  {p[k]:10.3f}")
