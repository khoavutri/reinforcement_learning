"""
================================================================================
 04 — ITERATIVE POLICY EVALUATION (DYNAMIC PROGRAMMING)   (Sutton & Barto, Ch. 4.1)
================================================================================

BÀI TOÁN
    Cho một MDP đã BIẾT mô hình p(s',r|s,a) và một chính sách π. Hãy tính
    hàm giá trị trạng thái v_π(s) — "nếu cứ đi theo π từ s thì được bao nhiêu
    phần thưởng (đã chiết khấu)?"

    Đây là bài toán DỰ ĐOÁN (prediction), không phải điều khiển.

Ý TƯỞNG
    Phương trình Bellman cho v_π:

        v_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [ r + γ v_π(s') ]

    Đây là hệ phương trình tuyến tính |S| ẩn. Với bài toán nhỏ có thể giải
    trực tiếp, nhưng cách tổng quát là LẶP: coi vế phải là một phép "kéo"
    v_π về phía chính nó, lặp đến khi hội tụ.

        V_{k+1}(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [ r + γ V_k(s') ]

    Mỗi vòng lặp như vậy gọi là một "sweep" (quét qua toàn bộ trạng thái).
    Phép toán này là một phép CO (contraction) với hệ số γ, nên chắc chắn
    hội tụ về v_π duy nhất, bất kể V_0.

PSEUDOCODE TRONG SÁCH
    Input: π — chính sách cần đánh giá
    Tham số: ngưỡng nhỏ θ > 0
    Khởi tạo V(s) tuỳ ý, V(terminal) = 0
    Lặp:
        Δ ← 0
        Với mỗi s ∈ S:
            v ← V(s)
            V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
            Δ ← max(Δ, |v - V(s)|)
    cho tới khi Δ < θ

CÁCH ĐỌC CODE
    - `V` là mảng nS phần tử; trạng thái kết thúc luôn giữ 0.
    - Vòng lặp trong cùng duyệt mọi hành động có xác suất > 0 dưới π.
    - Cập nhật TẠI CHỖ (in-place): dùng ngay giá trị vừa tính, hội tụ nhanh hơn
      bản dùng hai mảng (sách có bàn ở mục 4.1).
    - `Δ` là mức thay đổi lớn nhất trong một sweep — tiêu chuẩn dừng.

ĐỘ PHỨC TẠP
    Mỗi sweep: O(|S| · |A|). Số sweep đến hội tụ phụ thuộc γ (γ càng gần 1
    càng chậm). Với bài toán lớn, |S| là "lời nguyền số chiều" — đây chính là
    lý do phải chuyển sang Monte Carlo / TD ở các chương sau.

CHẠY THỬ
    python 04_policy_evaluation.py
================================================================================
"""

import numpy as np
from common import GridWorld, ARROWS

GAMMA = 0.9
THETA = 1e-6


def policy_evaluation(env, policy, gamma=GAMMA, theta=THETA, max_sweeps=10_000):
    """Ước lượng v_π cho chính sách `policy` (mảng nS, mỗi phần tử là 1 hành động)."""
    V = np.zeros(env.nS)

    for sweep in range(max_sweeps):
        delta = 0.0
        # quét toàn bộ trạng thái; bỏ qua trạng thái kết thúc (V luôn = 0)
        for s in range(env.nS):
            if env.is_terminal(s):
                continue

            v_old = V[s]
            # (1) v_π(s) = E[ r + γ V(s') ] dưới chính sách hiện tại
            #     Ở đây π là tất định nên chỉ có một hành động.
            a = policy[s]
            new_v = 0.0
            for prob, s2, r in env.transitions(s, a):
                new_v += prob * (r + gamma * V[s2])

            # (2) cập nhật tại chỗ
            V[s] = new_v

            # (3) ghi nhận mức thay đổi lớn nhất
            delta = max(delta, abs(v_old - new_v))

        if delta < theta:
            print(f"  hội tụ sau {sweep + 1} sweep (Δ = {delta:.2e})")
            break

    return V


def uniform_random_policy(env, seed=0):
    """Chính sách ngẫu nhiên đều — dùng để so sánh với chính sách tối ưu."""
    rng = np.random.default_rng(seed)
    return rng.integers(env.nA, size=env.nS)


if __name__ == "__main__":
    print("=" * 74)
    print(" ITERATIVE POLICY EVALUATION — lưới 4x4, đích ở góc dưới phải")
    print("=" * 74)

    env = GridWorld()
    policy = uniform_random_policy(env)

    print("\nChính sách đang đánh giá (mũi tên = hướng đi):")
    print(env.render(policy=policy))

    print("\nĐang lặp...")
    V = policy_evaluation(env, policy)

    print("\nGiá trị v_π (chính sách ngẫu nhiên):")
    print(env.render(V=V, policy=policy))

    print("\nGhi chú: giá trị giảm dần khi xa đích, đúng như trực giác:")
    print("càng nhiều bước thì càng bị trừ -1 nhiều lần, và bị chiết khấu γ^k.")
