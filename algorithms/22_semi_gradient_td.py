"""
================================================================================
 22 — SEMI-GRADIENT TD(0) VỚI XẤP XỈ HÀM TUYẾN TÍNH   (Sutton & Barto, Ch. 9.3)
================================================================================

BÀI TOÁN
    Trạng thái quá nhiều để lưu bảng. Thay V(s) bằng một hàm có tham số:

        v̂(s, w) = wᵀ x(s) = Σ_i w_i x_i(s)

    với x(s) là vector đặc trưng (feature) của trạng thái.

Ý TƯỞNG — SEMI-GRADIENT
    Ta muốn giảm sai số (v̂(S,w) − G)². Gradient thật là:

        ∇[1/2 (G − v̂)²] = −(G − v̂) ∇v̂

    Nhưng trong TD, "G" lại phụ thuộc vào w (qua v̂(S',w)). Nếu tính cả phần
    đó thì phức tạp và không ổn định. Semi-gradient chỉ lấy đạo hàm theo
    v̂(S,w), coi target là hằng số:

        w ← w + α [ R + γ v̂(S',w) − v̂(S,w) ] ∇v̂(S,w)

    Với tuyến tính, ∇v̂ = x(S), nên:

        w ← w + α δ x(S)          với δ = R + γ wᵀx(S') − wᵀx(S)

    "Semi" nghĩa là "nửa gradient" — không phải gradient thật, nhưng hội tụ
    tốt trong thực tế và là nền của mọi thuật toán deep RL.

PSEUDOCODE (sách, hộp 9.1)
    Input: π, hàm v̂ khả vi, α
    Khởi tạo w tuỳ ý
    Lặp với mỗi episode:
        khởi tạo S
        lặp mỗi bước:
            A ← π(·|S)
            thực hiện A, quan sát R, S'
            nếu S' terminal: w ← w + α[R − v̂(S,w)]∇v̂(S,w)
            ngược lại:       w ← w + α[R + γv̂(S',w) − v̂(S,w)]∇v̂(S,w)
            S ← S'

CÁCH ĐỌC CODE
    - `one_hot(s, n)` là bộ đặc trưng đơn giản nhất (tương đương bảng).
      Trong bài toán thật, thay bằng tile coding / Fourier / neural network.
    - `x_terminal = 0` để v̂(terminal) = 0 — đúng điều kiện biên.
    - Với one-hot, thuật toán này cho KẾT QUẢ Y HỆT TD(0) dạng bảng. Đó là
      cách tốt để kiểm chứng cài đặt trước khi đổi sang feature phức tạp.

CHẠY THỬ
    python 22_semi_gradient_td.py
================================================================================
"""

import numpy as np
from common import RandomWalk, one_hot


def semi_gradient_td(env, policy, episodes=100, alpha=0.05, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n_features = env.nS
    w = np.zeros(n_features)

    for _ in range(episodes):
        s = env.reset()
        x = one_hot(s, n_features)

        while not env.is_terminal(s):
            a = int(rng.choice(env.nA, p=policy))
            r, s2, done = env.step(s, a)

            x2 = np.zeros(n_features) if done else one_hot(s2, n_features)

            # (1) TD error với giá trị xấp xỉ
            delta = r + gamma * (w @ x2) - (w @ x)

            # (2) semi-gradient update: chỉ nhân với ∇v̂ = x
            w += alpha * delta * x

            x = x2
            s = s2

    return w


if __name__ == "__main__":
    print("=" * 68)
    print(" SEMI-GRADIENT TD(0) — RandomWalk 5 ô, feature one-hot")
    print("=" * 68)

    env = RandomWalk(n=5)
    policy = np.array([0.5, 0.5])
    truth = np.array([s / (env.n + 1) for s in range(env.nS)])

    print(f"\nGiá trị thật: {np.round(truth[1:-1], 3)}\n")
    print(f"{'alpha':>8}{'episode':>10}{'RMSE':>12}")
    print("-" * 30)

    for alpha in (0.01, 0.05, 0.2):
        for episodes in (50, 500):
            errs = []
            for seed in range(10):
                w = semi_gradient_td(env, policy, episodes=episodes, alpha=alpha, seed=seed)
                errs.append(np.sqrt(np.mean((w[1:-1] - truth[1:-1]) ** 2)))
            print(f"{alpha:>8}{episodes:>10}{np.mean(errs):>12.4f}")

    print("\nNhận xét: α quá nhỏ học chậm, α quá lớn dao động. Với one-hot,")
    print("kết quả trùng với TD(0) dạng bảng (file 10) — đúng như kỳ vọng.")
    print("Đổi `one_hot` thành tile coding là có ngay thuật toán cho bài toán lớn.")
