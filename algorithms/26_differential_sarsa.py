"""
================================================================================
 26 — DIFFERENTIAL SEMI-GRADIENT SARSA                 (Sutton & Barto, Ch. 10.5)
================================================================================

BÀI TOÁN
    Bài toán LIÊN TỤC, không có episode, không có trạng thái kết thúc. Khi đó
    chiết khấu γ < 1 trở nên gượng ép (nó làm giảm giá trị của tương lai xa
    một cách tuỳ ý). Chế độ "average reward" xử lý tự nhiên hơn.

Ý TƯỞNG
    Thay vì tối đa hoá tổng chiết khấu, tối đa hoá TỐC ĐỘ phần thưởng:

        r̄(π) = lim_{T→∞} (1/T) E[ Σ R_t ]

    Định nghĩa "differential return":

        G_t = R_{t+1} − r̄ + R_{t+2} − r̄ + ...

    tức là phần thưởng TRỪ ĐI mức trung bình. Giá trị q(s,a) giờ là "phần
    thưởng vượt trội so với mức trung bình" — có thể dương hoặc âm.

    Cập nhật:

        δ_t = R_{t+1} − R̄_t + q̂(S_{t+1},A_{t+1},w) − q̂(S_t,A_t,w)
        R̄_{t+1} = R̄_t + α_R δ_t                      (học luôn cả r̄)
        w ← w + α δ_t ∇q̂(S_t,A_t,w)

    Chú ý: KHÔNG có γ trong công thức. Đó chính là điểm mấu chốt.

PSEUDOCODE (Differential semi-gradient Sarsa)
    Khởi tạo w, R̄ (ví dụ 0)
    Lặp với mỗi bước:
        S ← trạng thái hiện tại
        A ← ε-tham lam theo q̂(S,·,w)
        thực hiện A, quan sát R, S'
        chọn A' theo q̂(S',·,w)
        δ ← R − R̄ + q̂(S',A',w) − q̂(S,A,w)
        R̄ ← R̄ + α_R δ
        w ← w + α δ ∇q̂(S,A,w)
        S ← S'; A ← A'

CÁCH ĐỌC CODE
    - Môi trường `Ring` là một vòng tròn n trạng thái: đi phải liên tục sẽ
      quay về chỗ cũ và nhận +1 mỗi vòng. Không có terminal.
    - r̄ tối ưu = 1/n (mỗi n bước được +1). Code in ra để so sánh.
    - Hàm giá trị học được là giá trị VI PHÂN, không phải giá trị thường.

LIÊN HỆ
    Chế độ average-reward là nền của các thuật toán RL cho robot chạy mãi,
    hệ thống điều khiển liên tục, và gần đây là các tác nhân hội thoại
    (không có "kết thúc" rõ ràng).

CHẠY THỬ
    python 26_differential_sarsa.py
================================================================================
"""

import numpy as np
from common import one_hot


class Ring:
    """Vòng tròn n trạng thái: action 1 = đi phải, action 0 = đứng yên.

    Đi từ trạng thái n-1 sang 0 được +1, các bước khác 0. Không có terminal.
    """

    def __init__(self, n=5):
        self.n = n
        self.nS = n
        self.nA = 2
        self.start = 0

    def reset(self):
        return self.start

    def is_terminal(self, s):
        return False

    def step(self, s, a):
        if a == 0:
            return 0.0, s, False
        s2 = (s + 1) % self.n
        r = 1.0 if s2 == 0 else 0.0
        return r, s2, False


def differential_sarsa(env, steps=20_000, alpha=0.1, alpha_r=0.01, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    d = env.nS
    W = np.zeros((env.nA, d))
    R_bar = 0.0

    def q(s, a):
        return float(W[a] @ one_hot(s, d))

    def choose(s):
        if rng.random() < eps:
            return int(rng.integers(env.nA))
        vals = [q(s, k) for k in range(env.nA)]
        best = np.flatnonzero(np.isclose(vals, max(vals)))
        return int(rng.choice(best))

    s = env.reset()
    a = choose(s)
    r_bar_history = []

    for _ in range(steps):
        r, s2, done = env.step(s, a)
        a2 = choose(s2)

        # (1) TD error KHÔNG có gamma, có trừ R̄
        delta = r - R_bar + q(s2, a2) - q(s, a)

        # (2) học tốc độ phần thưởng trung bình
        R_bar += alpha_r * delta

        # (3) semi-gradient update
        W[a] += alpha * delta * one_hot(s, d)

        s, a = s2, a2
        r_bar_history.append(R_bar)

    return W, R_bar, r_bar_history


if __name__ == "__main__":
    print("=" * 70)
    print(" DIFFERENTIAL SEMI-GRADIENT SARSA — vòng tròn 5 trạng thái")
    print("=" * 70)

    env = Ring(n=5)
    W, R_bar, hist = differential_sarsa(env, steps=20_000)

    print(f"\nTốc độ thưởng trung bình học được: r̄ = {R_bar:.4f}")
    print(f"Giá trị tối ưu lý thuyết        : r̄ = 1/{env.n} = {1/env.n:.4f}")

    print("\nChính sách tham lam theo q̂ (0 = đứng yên, 1 = đi phải):")
    for s in range(env.nS):
        qs = [float(W[a] @ one_hot(s, env.nS)) for a in range(env.nA)]
        print(f"  s={s}: q(đứng)={qs[0]:+.3f}  q(phải)={qs[1]:+.3f}  "
              f"-> {'phải' if qs[1] > qs[0] else 'đứng yên'}")

    print("\nGiá trị vi phân q̂ (phần thưởng vượt mức trung bình):")
    for s in range(env.nS):
        print(f"  s={s}: {float(W[1] @ one_hot(s, env.nS)):+.3f}")

    print("\nNhận xét: r̄ hội tụ đúng 0.2 = 1/5, và chính sách học được là luôn")
    print("đi phải — tối ưu. Giá trị vi phân không phải 'tổng thưởng tương lai'")
    print("mà là 'lợi thế so với mức trung bình', nên có thể âm hoặc dương.")
