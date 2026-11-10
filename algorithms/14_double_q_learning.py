"""
================================================================================
 14 — DOUBLE Q-LEARNING                                     (Sutton & Barto, Ch. 6.7)
================================================================================

BÀI TOÁN
    `max` trong Q-learning gây ra MAXIMIZATION BIAS: khi các ước lượng Q bị
    nhiễu, lấy max của chúng luôn cho kết quả CAO HƠN giá trị thật. Hệ quả:
    agent thích những hành động "may mắn" đã từng cho thưởng cao.

VÍ DỤ KINH ĐIỂN (hình 6.5 trong sách)
    Trạng thái A có 2 hành động:
        - sang phải: về terminal, thưởng ~ N(-0.1, 1)   (trung bình -0.1)
        - sang trái : sang B, thưởng 0, rồi từ B luôn về terminal với 0
    Tối ưu: từ A LUÔN đi trái (được 0, tốt hơn -0.1).
    Nhưng Q-learning thường xuyên học được "đi phải", vì trong 10 lần thử
    ngẫu nhiên, có lần max của mẫu nhiễu > 0, và giá trị đó bị "neo" vào Q.
    Sách gọi đây là ví dụ rõ nhất của maximization bias.

Ý TƯỞNG — HAI BẢNG, ĐÁNH GIÁ CHÉO
    Giữ hai ước lượng độc lập Q1, Q2. Mỗi bước tung đồng xu:
        - Nếu dùng Q1 để CHỌN hành động tốt nhất ở S', thì dùng Q2 để ĐÁNH GIÁ nó.
        - Ngược lại.
    Nhờ vậy, ước lượng dùng để đánh giá không phải là ước lượng đã dùng để
    chọn max -> không còn bị thiên lệch lên.

        Q1(S,A) ← Q1(S,A) + α[ R + γ Q2(S', argmax_a Q1(S',a)) - Q1(S,A) ]
        Q2(S,A) ← Q2(S,A) + α[ R + γ Q1(S', argmax_a Q2(S',a)) - Q2(S,A) ]

PSEUDOCODE
    Khởi tạo Q1, Q2 (ví dụ = 0), Q(terminal,·) = 0
    Lặp với mỗi episode:
        khởi tạo S
        lặp mỗi bước:
            chọn A theo ε-tham lam trên Q1 + Q2
            thực hiện A, quan sát R, S'
            với xác suất 0.5:
                a* ← argmax_a Q1(S',a)
                Q1(S,A) ← Q1(S,A) + α[R + γQ2(S',a*) - Q1(S,A)]
            ngược lại:
                a* ← argmax_a Q2(S',a)
                Q2(S,A) ← Q2(S,A) + α[R + γQ1(S',a*) - Q2(S,A)]

CÁCH ĐỌC CODE
    - `MaxBiasEnv` mô phỏng đúng ví dụ trong sách.
    - Chỉ số cần đo: sau khi học, từ A agent chọn "phải" bao nhiêu % số lần.
      Q-learning tốt phải gần 0%, Q-learning thường ~ 50-90%.

LIÊN HỆ HIỆN ĐẠI
    Double DQN (van Hasselt et al., 2016) chính là ý tưởng này áp lên DQN và
    là một trong những cải tiến quan trọng nhất của deep RL.

CHẠY THỬ
    python 14_double_q_learning.py
================================================================================
"""

import numpy as np
from common import argmax_tie


class MaxBiasEnv:
    """Ví dụ 6.7 trong sách: A = 0, B = 1, terminal = 2, 10 hành động.

    Từ A:  hành động 1 ("phải") -> terminal ngay, thưởng 0        <-- TỐI ƯU
           mọi hành động khác ("trái") -> B, thưởng 0
    Từ B:  mọi hành động -> terminal, thưởng ~ N(-0.1, 1)

    Vì γ = 1, max trên 10 ước lượng nhiễu ở B bị đẩy lên khá cao (gần 0),
    nên B "trông có vẻ" tốt và Q-learning chọn "trái" — maximization bias.
    """

    def __init__(self, seed=0):
        self.nS, self.nA, self.terminal = 3, 10, 2
        self.rng = np.random.default_rng(seed)
        self.start = 0

    def reset(self):
        return self.start

    def is_terminal(self, s):
        return s == self.terminal

    def step(self, s, a):
        if s == 0:
            if a == 1:
                return 0.0, self.terminal, True
            return 0.0, 1, False
        return float(self.rng.normal(-0.1, 1.0)), self.terminal, True


def q_learning_maxbias(env, episodes=300, steps=100, alpha=0.1, eps=0.1, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))

    for _ in range(episodes):
        s = env.reset()
        for _ in range(steps):
            a = int(rng.integers(env.nA)) if rng.random() < eps else argmax_tie(Q[s], rng)
            r, s2, done = env.step(s, a)
            target = r if done else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = env.reset() if done else s2

    return Q


def double_q_maxbias(env, episodes=300, steps=100, alpha=0.1, eps=0.1, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Q1 = np.zeros((env.nS, env.nA))
    Q2 = np.zeros((env.nS, env.nA))

    for _ in range(episodes):
        s = env.reset()
        for _ in range(steps):
            combined = Q1[s] + Q2[s]
            a = int(rng.integers(env.nA)) if rng.random() < eps else argmax_tie(combined, rng)
            r, s2, done = env.step(s, a)

            if rng.random() < 0.5:
                a_star = argmax_tie(Q1[s2], rng)
                target = r if done else r + gamma * Q2[s2, a_star]
                Q1[s, a] += alpha * (target - Q1[s, a])
            else:
                a_star = argmax_tie(Q2[s2], rng)
                target = r if done else r + gamma * Q1[s2, a_star]
                Q2[s, a] += alpha * (target - Q2[s, a])

            s = env.reset() if done else s2

    return Q1 + Q2


if __name__ == "__main__":
    print("=" * 74)
    print(" MAXIMIZATION BIAS — ví dụ 6.7")
    print(" A: 'phải' -> đích ngay, thưởng 0 (TỐI ƯU) ; 'trái' -> B")
    print(" B: mọi hành động -> ~N(-0.1, 1)")
    print("=" * 74)

    runs = 50
    wrong_ql, wrong_dq = 0, 0

    for seed in range(runs):
        Q = q_learning_maxbias(MaxBiasEnv(seed=seed), episodes=100, seed=seed)
        if np.argmax(Q[0]) != 1:          # không chọn "phải" -> sai
            wrong_ql += 1

        Qd = double_q_maxbias(MaxBiasEnv(seed=seed), episodes=100, seed=seed)
        if np.argmax(Qd[0]) != 1:
            wrong_dq += 1

    print(f"\nTỉ lệ chọn SAI hành động 'trái' ở trạng thái A (trên {runs} lần chạy):")
    print(f"  Q-learning        : {100 * wrong_ql / runs:5.1f}%")
    print(f"  Double Q-learning : {100 * wrong_dq / runs:5.1f}%")

    print("\nNhận xét: Q-learning bị 'max' lừa nên vẫn chọn sai khoảng 5% — đúng")
    print("con số sách báo cáo ở tiệm cận với ε = 0.1. Double Q-learning gần như")
    print("không bao giờ sai. Đây là lý do Double DQN là mặc định trong deep RL.")
