"""
================================================================================
 06 — VALUE ITERATION                                     (Sutton & Barto, Ch. 4.4)
================================================================================

BÀI TOÁN
    Như policy iteration: tìm π* và v*. Nhưng nhanh hơn bằng cách bỏ bước
    "đánh giá cho hội tụ hẳn".

Ý TƯỞNG
    Trong policy iteration, mỗi vòng ta chạy policy evaluation đến hội tụ
    (rất tốn). Nhưng nếu chỉ chạy ĐÚNG MỘT sweep đánh giá rồi cải thiện luôn
    thì sao? Khi đó hai bước gộp thành một phép cập nhật duy nhất:

        V_{k+1}(s) ← max_a Σ_{s',r} p(s',r|s,a) [ r + γ V_k(s') ]        (4.10)

    Chú ý: chỉ khác policy evaluation đúng một chỗ — thay Σ_a π(a|s) bằng max_a.
    Đây chính là phương trình Bellman TỐI ƯU được biến thành quy tắc cập nhật.

    Trực giác: value iteration = "mỗi vòng, giả sử mình sẽ hành động tối ưu
    từ bước sau trở đi". Sau nhiều vòng, giả định đó trở thành sự thật.

PSEUDOCODE TRONG SÁCH
    Tham số: ngưỡng θ > 0
    Khởi tạo V(s) tuỳ ý, V(terminal) = 0
    Lặp:
        Δ ← 0
        với mỗi s ∈ S:
            v ← V(s)
            V(s) ← max_a Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
            Δ ← max(Δ, |v - V(s)|)
    cho tới khi Δ < θ
    Xuất π(s) = argmax_a Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]

SO SÁNH VỚI POLICY ITERATION
    - Value iteration: mỗi vòng rẻ, cần nhiều vòng.
    - Policy iteration: mỗi vòng đắt, cần ít vòng.
    - Trong thực tế value iteration thường được ưa hơn vì đơn giản, dễ vector
      hoá, và không cần kiểm tra "chính sách có đổi không".

LƯU Ý VỀ HỘI TỤ
    V_k hội tụ về v* theo chuẩn max khi γ < 1. Số vòng cần thiết xấp xỉ
    O( log(1/ε) / (1-γ) ) — γ = 0.99 sẽ chậm hơn γ = 0.9 khoảng 10 lần.

CHẠY THỬ
    python 06_value_iteration.py
================================================================================
"""

import numpy as np
from common import GridWorld, ARROWS


def value_iteration(env, gamma=0.9, theta=1e-8, max_iters=10_000):
    V = np.zeros(env.nS)

    for it in range(1, max_iters + 1):
        delta = 0.0
        for s in range(env.nS):
            if env.is_terminal(s):
                continue

            v_old = V[s]
            # (1) Bellman optimality: lấy MAX thay vì trung bình theo π
            best = max(
                sum(p * (r + gamma * V[s2]) for p, s2, r in env.transitions(s, a))
                for a in range(env.nA)
            )
            V[s] = best

            # (2) theo dõi mức thay đổi lớn nhất
            delta = max(delta, abs(v_old - best))

        if delta < theta:
            print(f"  hội tụ sau {it} vòng (Δ = {delta:.2e})")
            break

    # (3) rút ra chính sách tham lam từ V
    policy = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        if env.is_terminal(s):
            continue
        values = [
            sum(p * (r + gamma * V[s2]) for p, s2, r in env.transitions(s, a))
            for a in range(env.nA)
        ]
        policy[s] = int(np.argmax(values))

    return policy, V


if __name__ == "__main__":
    print("=" * 74)
    print(" VALUE ITERATION — lưới 4x4, γ = 0.9")
    print("=" * 74)

    env = GridWorld(gamma=0.9)
    policy, V = value_iteration(env, gamma=0.9)

    print("\nGiá trị tối ưu v* (số) và hành động tối ưu (mũi tên):")
    print(env.render(V=V, policy=policy))

    print("\nKiểm chứng bằng tay ở ô (0,0):")
    print(f"  v*(0,0) = {V[0]:.4f}")
    print("  đi xuống hoặc sang phải đều tới đích sau 6 bước,")
    print("  5 bước đầu bị -1, bước cuối vào đích được 0,")
    print("  tổng chiết khấu = -(1 + 0.9 + ... + 0.9^4) = "
          f"{-sum(0.9 ** k for k in range(5)):.4f}")
