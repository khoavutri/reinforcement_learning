"""
common.py — Môi trường nhỏ dùng chung cho tất cả file thuật toán.

Mục đích: mỗi file thuật toán chỉ tập trung vào ĐÚNG phần thuật toán.
Các bài toán đồ chơi (gridworld, cliff walking, random walk, bandit)
được gom về đây để không phải lặp lại 30 lần.

Quy ước:
    - Trạng thái (state) là số nguyên (int) để dễ làm bảng V[s], Q[s][a].
    - Hành động (action) là số nguyên 0..nA-1.
    - step(s, a) -> (r, s2, done)   # giống gym: quan sát phần thưởng và trạng thái kế
    - transitions(s, a) -> [(xác suất, s2, r), ...]  # mô hình đầy đủ, dùng cho DP
"""

from __future__ import annotations

import numpy as np

# 4 hướng di chuyển: lên, xuống, trái, phải
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}


class GridWorld:
    """Lưới đơn giản: đi 4 hướng, mỗi bước -1, tới đích thì kết thúc.

    Dùng cho: policy evaluation, policy iteration, value iteration,
    TD, Monte Carlo, Sarsa, Q-learning...
    """

    def __init__(self, rows=4, cols=4, start=(0, 0), goal=(3, 3),
                 step_reward=-1.0, gamma=0.9, walls=()):
        self.rows, self.cols = rows, cols
        self.start_rc, self.goal_rc = start, goal
        self.step_reward = step_reward
        self.gamma = gamma
        self.walls = set(walls)          # các ô không đi vào được
        self.nS = rows * cols
        self.nA = 4
        self.terminal = self._to_state(goal)

    # --- chuyển đổi toạ độ <-> chỉ số -------------------------------------
    def _to_state(self, rc):
        return rc[0] * self.cols + rc[1]

    def _to_rc(self, s):
        return divmod(s, self.cols)

    # --- giao diện môi trường ---------------------------------------------
    def reset(self):
        return self._to_state(self.start_rc)

    def is_terminal(self, s):
        return s == self.terminal

    def step(self, s, a):
        if self.is_terminal(s):
            return 0.0, s, True

        r0, c0 = self._to_rc(s)
        dr, dc = MOVES[a]
        r1, c1 = r0 + dr, c0 + dc

        # đụng tường hoặc biên -> đứng yên
        if not (0 <= r1 < self.rows and 0 <= c1 < self.cols) or (r1, c1) in self.walls:
            r1, c1 = r0, c0

        s2 = self._to_state((r1, c1))
        if s2 == self.terminal:
            return 0.0, s2, True          # vào đích: hết episode, không phạt thêm
        return self.step_reward, s2, False

    def transitions(self, s, a):
        """Mô hình: với mỗi (s,a) liệt kê (xác suất, s', r). Ở đây tất định."""
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        r, s2, _ = self.step(s, a)
        return [(1.0, s2, r)]

    # --- hiển thị ----------------------------------------------------------
    def render(self, V=None, policy=None, title=""):
        """In lưới ra màn hình; V là dict/list giá trị, policy là dict/list hành động."""
        lines = []
        if title:
            lines.append(title)
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                s = self._to_state((r, c))
                if s == self.terminal:
                    row.append("   GOAL   ")
                elif (r, c) in self.walls:
                    row.append("   ####   ")
                elif V is not None:
                    mark = ARROWS[policy[s]] if policy is not None else " "
                    row.append(f"{mark}{V[s]:7.2f} ")
                elif policy is not None:
                    row.append(f"   {ARROWS[policy[s]]}    ")
                else:
                    row.append(f"   {r},{c}   ")
            lines.append(" ".join(row))
        return "\n".join(lines)


class CliffWalking(GridWorld):
    """Bài toán kinh điển ở mục 6.5: đi từ S tới G, rơi xuống vực thì -100 và về S.

    Dùng để so sánh Sarsa (an toàn) với Q-learning (đi sát mép vực).
    """

    def __init__(self, rows=4, cols=12):
        super().__init__(rows=rows, cols=cols,
                         start=(3, 0), goal=(3, cols - 1), step_reward=-1.0, gamma=1.0)
        self.cliff = {(3, c) for c in range(1, cols - 1)}

    def step(self, s, a):
        if self.is_terminal(s):
            return 0.0, s, True

        r0, c0 = self._to_rc(s)
        dr, dc = MOVES[a]
        r1, c1 = r0 + dr, c0 + dc
        if not (0 <= r1 < self.rows and 0 <= c1 < self.cols):
            r1, c1 = r0, c0

        if (r1, c1) in self.cliff:
            return -100.0, self._to_state(self.start_rc), False   # tụt vực: về vạch xuất phát

        s2 = self._to_state((r1, c1))
        if s2 == self.terminal:
            return -1.0, s2, True
        return -1.0, s2, False


class RandomWalk:
    """Chuỗi Markov 1 chiều n ô, đi trái/phải 50-50, thưởng +1 ở đầu phải.

    Dùng để minh hoạ TD so với Monte Carlo (mục 6.2, 7.1).
    """

    def __init__(self, n=5):
        self.n = n
        self.nS = n + 2          # thêm 2 trạng thái kết thúc ở hai đầu
        self.nA = 2
        self.start = (n + 1) // 2
        self.terminal = {0, n + 1}

    def reset(self):
        return self.start

    def is_terminal(self, s):
        return s in self.terminal

    def step(self, s, a):
        s2 = s + (1 if a == 1 else -1)
        if s2 in self.terminal:
            return (1.0 if s2 == self.n + 1 else 0.0), s2, True
        return 0.0, s2, False

    def transitions(self, s, a):
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        r, s2, _ = self.step(s, a)
        return [(1.0, s2, r)]


class Corridor:
    """Hành lang tất định: 2 hành động trái/phải, +1 khi tới đầu phải.

    Giống RandomWalk nhưng agent ĐIỀU KHIỂN được hướng đi, nên dùng được
    cho các thuật toán control (ví dụ Monte Carlo ES ở chương 5).
    """

    def __init__(self, n=5, step_reward=0.0):
        self.n = n
        self.nS = n + 2
        self.nA = 2
        self.start = (n + 1) // 2
        self.step_reward = step_reward
        self.terminal = {0, n + 1}

    def reset(self):
        return self.start

    def is_terminal(self, s):
        return s in self.terminal

    def step(self, s, a):
        if self.is_terminal(s):
            return 0.0, s, True
        s2 = s + (1 if a == 1 else -1)
        if s2 in self.terminal:
            return (1.0 if s2 == self.n + 1 else 0.0), s2, True
        return self.step_reward, s2, False

    def transitions(self, s, a):
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        r, s2, _ = self.step(s, a)
        return [(1.0, s2, r)]


class Bandit:
    """Máy đánh bạc k tay: mỗi tay cho phần thưởng ~ N(q*, 1)."""

    def __init__(self, k=10, seed=0):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_true = self.rng.normal(0.0, 1.0, k)   # giá trị thật, ẩn với agent
        self.best = int(np.argmax(self.q_true))

    def step(self, a):
        return self.q_true[a] + self.rng.normal()


# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------
def argmax_tie(values, rng=None):
    """argmax có phá hoà ngẫu nhiên (sách nhấn mạnh: phải phá hoà nhất quán)."""
    values = np.asarray(values, dtype=float)
    best = np.flatnonzero(values == values.max())
    if rng is None or len(best) == 1:
        return int(best[0])
    return int(rng.choice(best))


def epsilon_greedy(Q, eps, rng):
    """Chọn hành động ε-tham lam. Q là mảng 1 chiều (giá trị từng hành động)."""
    if rng.random() < eps:
        return int(rng.integers(len(Q)))
    return argmax_tie(Q, rng)


def softmax(z):
    """Chuyển điểm ưa thích (preference) thành xác suất; trừ max để khỏi tràn số."""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def one_hot(i, n):
    """Vector đặc trưng one-hot: dùng làm hàm xấp xỉ tuyến tính đơn giản nhất."""
    x = np.zeros(n)
    x[i] = 1.0
    return x


def print_header(title):
    print("=" * 70)
    print(title)
    print("=" * 70)
