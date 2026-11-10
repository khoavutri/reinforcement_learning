"""
================================================================================
 23 — EPISODIC SEMI-GRADIENT SARSA                      (Sutton & Barto, Ch. 10.1)
================================================================================

BÀI TOÁN
    Điều khiển (tìm π*) khi trạng thái quá nhiều để lưu bảng, dùng xấp xỉ
    hàm cho q̂(s,a,w).

Ý TƯỞNG
    Áp semi-gradient (file 22) lên Sarsa (file 11):

        w ← w + α [ R + γ q̂(S',A',w) − q̂(S,A,w) ] ∇q̂(S,A,w)

    Với xấp xỉ tuyến tính theo từng hành động:

        q̂(s,a,w) = w_aᵀ x(s)

    thì ∇q̂ = x(s) chỉ cho khối w_a (các hành động khác không đổi).

    Chính sách: ε-tham lam theo q̂(·,·,w).

PSEUDOCODE (Episodic Semi-gradient Sarsa)
    Input: hàm q̂ khả vi, α, ε
    Khởi tạo w
    Lặp với mỗi episode:
        khởi tạo S
        chọn A ~ ε-tham lam theo q̂(S,·,w)
        lặp mỗi bước:
            thực hiện A, quan sát R, S'
            nếu S' terminal:
                w ← w + α[R − q̂(S,A,w)]∇q̂(S,A,w)
            ngược lại:
                chọn A' ~ ε-tham lam theo q̂(S',·,w)
                w ← w + α[R + γq̂(S',A',w) − q̂(S,A,w)]∇q̂(S,A,w)
            S ← S'; A ← A'

CÁCH ĐỌC CODE
    - `W` là ma trận (nA × nFeatures). `q(s,a)` = W[a] @ x(s).
    - `x` chỉ là one-hot nên bài này tương đương Sarsa dạng bảng; nhưng cấu
      trúc code đã sẵn sàng cho feature phức tạp (chỉ cần đổi `phi`).
    - `phi(s)` là hàm đặc trưng — điểm duy nhất cần thay khi nâng cấp.

LƯU Ý LÝ THUYẾT
    Với xấp xỉ hàm + off-policy + bootstrapping = DEADLY TRIAD (chương 11)
    và có thể phân kỳ. Sarsa là on-policy nên an toàn hơn trong trường hợp này.

CHẠY THỬ
    python 23_semi_gradient_sarsa.py
================================================================================
"""

import numpy as np
from common import GridWorld, one_hot


def semi_gradient_sarsa(env, episodes=50, alpha=0.1, gamma=0.9, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_features = env.nS
    W = np.zeros((env.nA, n_features))

    def phi(s):
        return np.zeros(n_features) if env.is_terminal(s) else one_hot(s, n_features)

    def q(x, a):
        return float(W[a] @ x)

    def choose(x):
        if rng.random() < eps:
            return int(rng.integers(env.nA))
        values = np.array([q(x, a) for a in range(env.nA)])
        best = np.flatnonzero(values == values.max())
        return int(rng.choice(best))

    for _ in range(episodes):
        s = env.reset()
        x = phi(s)
        a = choose(x)

        while True:
            r, s2, done = env.step(s, a)
            x2 = phi(s2)

            # (1) target: nếu kết thúc thì chỉ có R
            if done:
                delta = r - q(x, a)
            else:
                a2 = choose(x2)
                delta = r + gamma * q(x2, a2) - q(x, a)

            # (2) cập nhật khối trọng số của hành động A
            W[a] += alpha * delta * x

            if done:
                break
            s, a, x = s2, a2, x2

    return W


def steps_to_goal(env, W, cap=200):
    def q(s, a):
        return float(W[a] @ one_hot(s, env.nS)) if not env.is_terminal(s) else 0.0

    s, n = env.reset(), 0
    while not env.is_terminal(s) and n < cap:
        a = max(range(env.nA), key=lambda k: q(s, k))
        r, s2, done = env.step(s, a)
        s = s2
        n += 1
    return n if env.is_terminal(s) else cap


if __name__ == "__main__":
    print("=" * 70)
    print(" EPISODIC SEMI-GRADIENT SARSA — lưới 4x4, TB 10 lần chạy")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, goal=(3, 3), gamma=0.9)
    print(f"\n{'episode':>8}{'số bước tới đích':>20}{'tối ưu':>12}")
    print("-" * 42)
    for episodes in (10, 25, 50, 100):
        scores = []
        for seed in range(10):
            W = semi_gradient_sarsa(env, episodes=episodes, alpha=0.5, seed=seed)
            scores.append(steps_to_goal(env, W))
        print(f"{episodes:>8}{np.mean(scores):>20.1f}{6:>12}")

    print("\nNhận xét: càng nhiều episode, chính sách càng gần đường ngắn nhất")
    print("(6 bước từ góc trên trái tới góc dưới phải của lưới 4x4).")
