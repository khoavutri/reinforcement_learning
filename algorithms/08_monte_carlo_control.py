"""
================================================================================
 08 — MONTE CARLO CONTROL: EXPLORING STARTS & ε-SOFT     (Sutton & Barto, Ch. 5.3–5.4)
================================================================================

BÀI TOÁN
    Tìm chính sách tối ưu khi KHÔNG biết mô hình. Monte Carlo cho ta q_π,
    nhưng để cải thiện π thành tham lam thì cần q của MỌI hành động — kể cả
    hành động chưa bao giờ được thử. Làm sao đảm bảo có dữ liệu cho tất cả?

HAI CÁCH GIẢI QUYẾT CỦA SÁCH

  (A) EXPLORING STARTS (mục 5.3)
      Mỗi episode bắt đầu từ một cặp (s, a) CHỌN NGẪU NHIÊN, rồi mới đi theo π.
      Nhờ vậy mọi cặp đều được thử ít nhất một lần.

      Vấn đề: hoàn toàn không thực tế. Không ai bắt được robot khởi động lại
      ở mọi trạng thái tuỳ ý. Sách nói thẳng điều này.

  (B) CHÍNH SÁCH ε-MỀM (mục 5.4)
      Không cần exploring starts. Chỉ cần π luôn giữ xác suất > 0 cho mọi
      hành động:
          π(a|s) ≥ ε/|A(s)|   với mọi a
      Khi đó mọi cặp (s,a) đều được thăm với xác suất 1, và ta học được
      chính sách ε-tối ưu (tốt nhất trong họ chính sách ε-mềm).

CÔNG THỨC (ε-soft)
    Từ Q, chính sách ε-tham lam:
        π(a|s) = 1 - ε + ε/|A|   nếu a = argmax Q(s,·)
        π(a|s) = ε/|A|           ngược lại

    Cập nhật Monte Carlo first-visit cho cặp (s,a), rồi cải thiện π.

PSEUDOCODE (ε-soft, mục 5.4)
    Khởi tạo Q(s,a) tuỳ ý, π là ε-mềm theo Q
    Lặp mãi:
        sinh episode theo π: S0,A0,R1,...,S_{T-1},A_{T-1},R_T
        G ← 0
        với t = T-1 ... 0:
            G ← γG + R_{t+1}
            nếu (S_t,A_t) chưa gặp trong episode:
                thêm G vào Returns(S_t,A_t)
                Q(S_t,A_t) ← trung bình(Returns(S_t,A_t))
                với mọi a: π(a|S_t) ← ε-mềm theo Q(S_t,·)

CÁCH ĐỌC CODE
    - `mc_exploring_starts`: bản trung thực của sách. Chạy trên hành lang
      1 chiều (Corridor) để chính sách tham lam luôn kết thúc được episode.
    - `mc_epsilon_soft`: bản dùng được trong thực tế, chạy trên lưới 4x4.
    - `max_steps`: chặn episode vô hạn do chính sách tham lam bị kẹt vòng lặp
      (một chi tiết kỹ thuật mà pseudocode trong sách bỏ qua).

CHẠY THỬ
    python 08_monte_carlo_control.py
================================================================================
"""

import numpy as np
from common import GridWorld, Corridor, argmax_tie


def mc_exploring_starts(env, episodes=20_000, gamma=1.0, max_steps=200, seed=0):
    """Monte Carlo ES — đúng nguyên văn pseudocode trong sách (mục 5.3)."""
    rng = np.random.default_rng(seed)

    Q = np.zeros((env.nS, env.nA))
    N = np.zeros((env.nS, env.nA))
    policy = np.zeros(env.nS, dtype=int)

    non_terminal = [s for s in range(env.nS) if not env.is_terminal(s)]

    for _ in range(episodes):
        # (1) exploring start: chọn ngẫu nhiên cặp (s, a)
        s = int(rng.choice(non_terminal))
        a = int(rng.integers(env.nA))

        states, actions, rewards = [s], [a], []
        done, steps = False, 0
        while not done and steps < max_steps:
            r, s2, done = env.step(s, a)
            rewards.append(r)
            s = s2
            steps += 1
            if not done:
                a = int(policy[s])          # sau đó đi theo π (tham lam)
                states.append(s)
                actions.append(a)

        if not done:
            continue                        # episode bị cắt: bỏ, không học

        # (2) duyệt ngược, first-visit cho từng cặp (s,a)
        G = 0.0
        seen = set()
        for t in range(len(rewards) - 1, -1, -1):
            G = gamma * G + rewards[t]
            sa = (states[t], actions[t])
            if sa in seen:
                continue
            seen.add(sa)

            st, at = sa
            N[st, at] += 1
            Q[st, at] += (G - Q[st, at]) / N[st, at]

            # (3) cải thiện chính sách ngay tại chỗ
            policy[st] = argmax_tie(Q[st], rng)

    return policy, Q


def mc_epsilon_soft(env, episodes=20_000, gamma=0.9, eps=0.1, max_steps=200, seed=0):
    """Monte Carlo control với chính sách ε-mềm (mục 5.4) — bản dùng thực tế."""
    rng = np.random.default_rng(seed)

    Q = np.zeros((env.nS, env.nA))
    N = np.zeros((env.nS, env.nA))

    def policy_probs(s):
        p = np.full(env.nA, eps / env.nA)
        p[int(np.argmax(Q[s]))] += 1.0 - eps
        return p

    for _ in range(episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        done, steps = False, 0
        while not done and steps < max_steps:
            a = int(rng.choice(env.nA, p=policy_probs(s)))
            r, s2, done = env.step(s, a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s2
            steps += 1

        if not done:
            continue

        G = 0.0
        seen = set()
        for t in range(len(rewards) - 1, -1, -1):
            G = gamma * G + rewards[t]
            sa = (states[t], actions[t])
            if sa in seen:
                continue
            seen.add(sa)

            st, at = sa
            N[st, at] += 1
            Q[st, at] += (G - Q[st, at]) / N[st, at]
            # π tự động trở thành ε-tham lam theo Q mới — không cần biến π riêng

    policy = np.argmax(Q, axis=1)
    return policy, Q


if __name__ == "__main__":
    print("=" * 70)
    print(" (A) MONTE CARLO ES — hành lang 5 ô (mọi cặp (s,a) đều được thử)")
    print("=" * 70)
    cor = Corridor(n=5, step_reward=-0.1)
    policy, Q = mc_exploring_starts(cor, episodes=3000)
    print("Q(s, trái/phải):")
    for s in range(1, cor.n + 1):
        print(f"  s={s}: {Q[s, 0]:+.3f} / {Q[s, 1]:+.3f}   -> chọn "
              f"{'phải' if policy[s] == 1 else 'trái'}")
    print("Học được: luôn đi sang phải để nhận +1. Đúng tối ưu.")

    print()
    print("=" * 70)
    print(" (B) MONTE CARLO ε-SOFT — lưới 4x4, ε = 0.1, 20.000 episode")
    print("=" * 70)
    env = GridWorld(gamma=0.9)
    policy, Q = mc_epsilon_soft(env, episodes=20_000)
    V = Q.max(axis=1)
    V[env.terminal] = 0.0
    print(env.render(V=V, policy=policy))
    print("\nSo với đáp án chính xác: v*(0,0) = -4.095 ; v*(3,0) = -1.900")
    print(f"Monte Carlo ước lượng:   v(0,0) = {V[0]:.3f}  ; v(12) = {V[12]:.3f}")
    print("\nSai lệch nhỏ là do ε = 0.1: agent vẫn thỉnh thoảng đi sai,")
    print("nên giá trị học được là của chính sách ε-mềm, không phải π* thuần.")
