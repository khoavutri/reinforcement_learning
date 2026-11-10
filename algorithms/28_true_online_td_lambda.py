"""
================================================================================
 28 — TRUE ONLINE TD(λ)                                 (Sutton & Barto, Ch. 12.5)
================================================================================

BÀI TOÁN
    TD(λ) ở file 27 dùng "accumulating trace" và cập nhật online — nhưng nó
    KHÔNG tương đương chính xác với λ-return khi các trọng số thay đổi trong
    episode. Nếu ta muốn kết quả trùng khớp với "online λ-return" (một thuật
    toán lý thuyết đẹp), cần một dạng trace khác.

Ý TƯỞNG — DUTCH TRACE
    Dùng "dutch trace" thay cho accumulating trace:

        z_t = γλ z_{t-1} + (1 − α γ λ (z_{t-1}ᵀ x_t)) x_t
        w   ← w + α [ δ_t + v̂(S_t,w) − v̂_old(S_t,w) ] z_t
                − α [ v̂(S_t,w) − v̂_old(S_t,w) ] x_t

    Số hạng hiệu chỉnh `(v̂ − v̂_old)` bù cho việc w đã thay đổi ngay trong
    bước. Kết quả: true online TD(λ) cho ra CHÍNH XÁC cùng trọng số với
    online λ-return, tại mọi thời điểm — không chỉ ở giới hạn.

    Đây là thuật toán "đúng" về lý thuyết, và cũng là nền của True Online
    Sarsa(λ) — một trong những thuật toán mạnh nhất của RL cổ điển.

PSEUDOCODE (sách, hộp 12.3)
    Khởi tạo w, z = 0
    khởi tạo x, V_old = 0
    lặp mỗi bước:
        A ~ π; thực hiện A, quan sát R, x'
        V ← wᵀx
        V' ← wᵀx'
        δ ← R + γV' − V
        z ← γλz + (1 − αγλ zᵀx) x
        w ← w + α(δ + V − V_old) z − α(V − V_old) x
        V_old ← V'
        x ← x'

CÁCH ĐỌC CODE
    - `V_old` là giá trị của trạng thái hiện tại được tính ở bước TRƯỚC
      (trước khi w thay đổi). Đây là biến dễ bị bỏ sót nhất.
    - Có 3 dòng cập nhật (z, w, V_old) — nhưng tất cả là O(d).
    - Với one-hot, kết quả rất gần TD(λ); khác biệt rõ hơn khi dùng feature
      chồng lấn (tile coding, RBF) hoặc α lớn.

CHẠY THỬ
    python 28_true_online_td_lambda.py
================================================================================
"""

import numpy as np
from common import RandomWalk, one_hot


def true_online_td_lambda(env, policy, episodes=10, alpha=0.05, lam=0.8,
                          gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    d = env.nS
    w = np.zeros(d)

    for _ in range(episodes):
        x = one_hot(env.reset(), d)
        z = np.zeros(d)
        V_old = 0.0

        s = int(np.argmax(x))
        while not env.is_terminal(s):
            a = int(rng.choice(env.nA, p=policy))
            r, s2, done = env.step(s, a)
            x2 = np.zeros(d) if done else one_hot(s2, d)

            V = w @ x
            V_next = w @ x2
            delta = r + gamma * V_next - V

            # (1) dutch trace: có hiệu chỉnh theo αγλ(zᵀx)
            z = gamma * lam * z + (1.0 - alpha * gamma * lam * (z @ x)) * x

            # (2) cập nhật kèm số hạng hiệu chỉnh (V - V_old)
            w += alpha * (delta + V - V_old) * z - alpha * (V - V_old) * x

            # (3) chuẩn bị cho bước sau
            V_old = V_next
            x = x2
            s = s2

    return w


if __name__ == "__main__":
    print("=" * 68)
    print(" TRUE ONLINE TD(λ) — RandomWalk 5 ô, α=0.05")
    print("=" * 68)

    env = RandomWalk(n=5)
    policy = np.array([0.5, 0.5])
    truth = np.array([s / (env.n + 1) for s in range(env.nS)])

    print(f"\n{'λ':>6}{'RMSE (5 ep)':>18}{'RMSE (20 ep)':>18}")
    print("-" * 42)
    for lam in (0.0, 0.4, 0.8, 1.0):
        e5, e20 = [], []
        for seed in range(20):
            w = true_online_td_lambda(env, policy, episodes=5, lam=lam, seed=seed)
            e5.append(np.sqrt(np.mean((w[1:-1] - truth[1:-1]) ** 2)))
            w = true_online_td_lambda(env, policy, episodes=20, lam=lam, seed=seed)
            e20.append(np.sqrt(np.mean((w[1:-1] - truth[1:-1]) ** 2)))
        print(f"{lam:>6}{np.mean(e5):>18.4f}{np.mean(e20):>18.4f}")

    print("\nNhận xét: true online TD(λ) học rất nhanh ở λ trung bình, và không")
    print("bị lệch so với λ-return lý thuyết. Đây là bản 'đúng' của TD(λ).")
