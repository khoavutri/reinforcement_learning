# BÁO CÁO TÓM TẮT SÁCH
## *Reinforcement Learning: An Introduction* — Richard S. Sutton & Andrew G. Barto
### (2nd edition, MIT Press, 2018 — bản in 2020)

---

## 0. Thông tin chung

| Mục | Nội dung |
|---|---|
| Tên sách | Reinforcement Learning: An Introduction (Học tăng cường: Nhập môn) |
| Tác giả | Richard S. Sutton (ĐH Alberta, DeepMind) và Andrew G. Barto (ĐH Massachusetts Amherst) |
| Nhà xuất bản | MIT Press, Cambridge, Massachusetts |
| Lần xuất bản | Thứ 2, 2018 (bản PDF đang dùng in năm 2020) |
| Số chương | 17 chương, chia 3 phần + thư mục tham khảo |
| Lĩnh vực | Học máy, trí tuệ nhân tạo, khoa học nhận thức, điều khiển học |
| Đối tượng | Sinh viên năm cuối / cao học ngành AI, kỹ sư ML, người nghiên cứu |

Đây là cuốn sách **giáo khoa chuẩn mực (canonical)** của ngành học tăng cường. Gần như mọi
khoá RL trên thế giới (kể cả khoá của David Silver tại DeepMind, khoá CS234 của Stanford,
Berkeley CS285) đều lấy nó làm nền. Cuốn sách không dạy "dùng thư viện nào", mà dạy
**vì sao thuật toán hoạt động** — từ công thức Bellman cho tới lý do Q-learning phân kỳ
khi ghép với xấp xỉ hàm.

Sách có 3 phần, đi theo đúng lộ trình mà một người học nên đi:

1. **Phần I — Tabular Solution Methods (Ch. 2–8):** mọi thứ đều là *bảng*. Trạng thái ít,
   ta lưu V(s), Q(s,a) vào mảng. Đây là phần "hiểu bản chất nhất", chiếm gần nửa sách.
2. **Phần II — Approximate Solution Methods (Ch. 9–13):** trạng thái nhiều vô kể, phải
   xấp xỉ bằng hàm (tuyến tính, neural network), thêm eligibility trace và policy gradient.
3. **Phần III — Looking Deeper (Ch. 14–17):** nhìn sang tâm lý học, khoa học thần kinh,
   ứng dụng thực tế và các biên giới nghiên cứu.

---

## 1. Học tăng cường là gì?

### 1.1. Định nghĩa một câu

> Học tăng cường (RL) là bài toán **một agent phải học cách hành động trong môi trường
> để tối đa hoá tổng phần thưởng nhận được về lâu dài**, chỉ dựa trên kinh nghiệm thử–sai,
> không ai dạy nó "đáp án đúng" là gì.

Ba điểm khác biệt với học có giám sát (supervised learning):

| | Học có giám sát | Học tăng cường |
|---|---|---|
| Dữ liệu | Tập (x, y) có nhãn sẵn | Agent tự sinh dữ liệu bằng cách hành động |
| Tín hiệu | Nhãn đúng cho từng mẫu | Chỉ có **phần thưởng**, thường trễ, thưa, nhiễu |
| Mục tiêu | Đoán đúng trên phân phối mẫu | Hành vi tối ưu **dài hạn**, có hệ quả ngược lại môi trường |
| Thời gian | Mẫu độc lập | Mẫu phụ thuộc nhau, hành động hôm nay đổi dữ liệu ngày mai |

### 1.2. Vòng lặp agent – môi trường

```
        ┌──────────────┐   hành động A_t    ┌──────────────┐
        │    AGENT     │ ─────────────────► │  MÔI TRƯỜNG   │
        │  (chính sách,│                    │  (dynamics,  │
        │   giá trị)   │ ◄───────────────── │   phần thưởng)│
        └──────────────┘  R_{t+1}, S_{t+1}  └──────────────┘
```

Tại mỗi bước thời gian t: agent ở trạng thái `S_t`, chọn hành động `A_t`, môi trường trả về
phần thưởng `R_{t+1}` và trạng thái mới `S_{t+1}`. Chuỗi này lặp lại tạo thành **episode**
(nếu có điểm kết thúc) hoặc chạy mãi (nếu là bài toán liên tục).

### 1.3. Bốn thành phần của một hệ RL

Sách liệt kê 4 thành phần, trong đó 3 cái đầu là **bắt buộc**, cái thứ 4 là tuỳ chọn:

1. **Chính sách (policy) π(a|s)** — cách agent quyết định. Có thể là bảng tra cứu
   (deterministic) hoặc phân phối xác suất (stochastic).
2. **Tín hiệu phần thưởng (reward signal) R** — mục tiêu tức thời. Đây là thứ **định nghĩa
   bài toán**, không phải thứ agent được phép sửa. Viết sai reward là hỏng cả hệ thống.
3. **Hàm giá trị (value function) v(s), q(s,a)** — dự đoán tổng phần thưởng *về sau*.
   Đây là điểm tinh tế nhất: reward nói "cái gì tốt ngay bây giờ", value nói "cái gì tốt
   về lâu dài". Hầu hết thuật toán RL là các cách khác nhau để **học hàm giá trị**.
4. **Mô hình (model)** — cái agent tự tưởng tượng về môi trường (dự đoán `s'`, `r`).
   Có model ⇒ **model-based** (planning, Dyna, MCTS). Không có ⇒ **model-free**.

### 1.4. Đánh đổi trung tâm: khám phá vs khai thác

**Exploration vs Exploitation.** Nếu chỉ làm điều đã biết là tốt, agent sẽ kẹt ở phương án
tầm thường. Nếu chỉ thử cái mới, nó không bao giờ kiếm được gì. Cuốn sách dành hẳn Chương 2
cho bài toán này dưới dạng thuần khiết nhất (multi-armed bandit) — nơi không có trạng thái,
chỉ có kéo tay và nhận thưởng — để thấy rõ bản chất của "cái giá của việc khám phá".

---

## 2. Nền tảng toán học: MDP (Chương 3)

Gần như toàn bộ sách xoay quanh **quá trình quyết định Markov** (Markov Decision Process).
Một MDP gồm:

- Tập trạng thái `S`, tập hành động `A`
- Xác suất chuyển `p(s', r | s, a)` — "làm a ở s, thì sang s' và được r với xác suất bao nhiêu"
- Hệ số chiết khấu `γ ∈ [0,1]`

**Tính Markov** nghĩa là: tương lai chỉ phụ thuộc trạng thái hiện tại, không phụ thuộc quá khứ.
Đây là giả định mạnh, và cũng là lý do lớn nhất khiến RL khó áp dụng vào bài toán thực:
trạng thái quan sát được thường **không đủ** (đó là POMDP).

### 2.1. Return và chiết khấu

Mục tiêu không phải `R_{t+1}` mà là **tổng có chiết khấu**:

```
G_t = R_{t+1} + γ·R_{t+2} + γ²·R_{t+3} + ... = Σ_{k=0..∞} γ^k · R_{t+k+1}
```

`γ` gần 1 ⇒ agent quan tâm dài hạn; `γ` gần 0 ⇒ agent "ăn xổi". Về mặt toán học, `γ < 1`
còn đảm bảo chuỗi hội tụ khi phần thưởng bị chặn.

### 2.2. Hai hàm giá trị và hai phương trình Bellman

```
v_π(s) = E_π[ G_t | S_t = s ]                    (giá trị trạng thái)
q_π(s,a) = E_π[ G_t | S_t = s, A_t = a ]         (giá trị hành động)
```

**Phương trình Bellman kỳ vọng** — chìa khoá của cả cuốn sách: giá trị của hiện tại
= thưởng tức thời + giá trị chiết khấu của tương lai.

```
v_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [ r + γ·v_π(s') ]
```

**Phương trình Bellman tối ưu** — khi được chọn hành động tốt nhất:

```
v*(s)  = max_a Σ_{s',r} p(s',r|s,a) [ r + γ·v*(s') ]
q*(s,a) = Σ_{s',r} p(s',r|s,a) [ r + γ·max_{a'} q*(s',a') ]
```

Mọi thuật toán trong Phần I chỉ là các cách **xấp xỉ lời giải** của hai phương trình này:
- Biết model ⇒ lặp trực tiếp (Dynamic Programming, Ch.4)
- Không biết model ⇒ lấy mẫu (Monte Carlo Ch.5, TD Ch.6)

`v*` và `q*` có quan hệ: một chính sách **tham lam theo `v*`** là tối ưu. Và với `q*` thì
không cần model: `π*(s) = argmax_a q*(s,a)`. Đây là lý do các thuật toán điều khiển
(control) thường học `q` chứ không học `v`.

---

## 3. Tóm tắt từng chương

### Chương 1 — Introduction
Định nghĩa RL, so sánh với học có giám sát, liệt kê 4 thành phần. Ví dụ xuyên suốt là
**tic-tac-toe**: một agent học bằng cách chơi với chính nó, cập nhật giá trị trạng thái
theo "giá trị trạng thái sau khi đi nước tốt nhất". Điểm cần nhớ: ví dụ này đã chứa đủ
mầm mống của mọi ý tưởng sau này (bootstrapping, khám phá, hàm giá trị). Cuối chương là
lịch sử RL: tâm lý học thử–sai (Thorndike), điều khiển tối ưu (Bellman), TD learning
(Sutton 1988).

### Chương 2 — Multi-armed Bandits
Bài toán "k tay máy": mỗi lần chọn 1 tay, nhận thưởng ngẫu nhiên. Không có trạng thái,
chỉ có đánh đổi khám phá/khai thác. Các thuật toán:

- **ε-greedy**: với xác suất ε chọn ngẫu nhiên, còn lại chọn tay tốt nhất hiện biết.
  Đơn giản, hiệu quả, vẫn là baseline mặc định trong thực tế.
- **Khởi tạo lạc quan (optimistic initial values)**: khởi tạo Q cao hơn thực tế ⇒ agent
  buộc phải thử hết mọi tay. Mẹo "khám phá mà không cần tham số ε", nhưng chỉ dùng được
  khi bài toán dừng (stationary).
- **UCB (Upper Confidence Bound)**: chọn theo `Q(a) + c·√(ln t / N(a))`. Tay nào ít được
  thử thì phần "thưởng thêm" càng lớn ⇒ khám phá có định hướng, không lãng phí.
- **Gradient bandit**: không ước lượng giá trị mà học **điểm ưa thích** `H(a)` bằng
  gradient ascent trên xác suất softmax. Đây là tổ tiên của policy gradient (Chương 13).

Bài học lớn: **giá trị ước lượng luôn có sai số, và khám phá là cách mua thông tin bằng
phần thưởng ngắn hạn.** Cập nhật trung bình tăng dần `Q ← Q + α[R − Q]` xuất hiện ở đây
và sẽ theo suốt cuốn sách.

### Chương 3 — Finite Markov Decision Processes
Định hình toàn bộ ngôn ngữ của sách: MDP, return, chiết khấu, `v_π`, `q_π`, phương trình
Bellman, chính sách tối ưu, hàm giá trị tối ưu. Kết thúc bằng mục "Optimality and
Approximation": trong bài toán lớn, ta **không bao giờ tìm được tối ưu thật**, chỉ tìm
được xấp xỉ đủ tốt — đây là chủ đề của cả Phần II.

### Chương 4 — Dynamic Programming
Khi **đã biết model** `p(s',r|s,a)`, có thể tính chính xác bằng quy hoạch động:

- **Policy evaluation**: cho π, lặp `V(s) ← Σ π(a|s) Σ p [r + γV(s')]` tới khi hội tụ.
- **Policy improvement**: thay π bằng chính sách tham lam theo V vừa tính.
- **Policy iteration**: lặp evaluation ↔ improvement tới khi π không đổi.
- **Value iteration**: gộp hai bước, mỗi vòng chỉ một sweep: `V(s) ← max_a Σ p[r + γV(s')]`.

Điểm hay nhất chương này là khái niệm **GPI (Generalized Policy Iteration)**: hai quá trình
đánh giá và cải thiện chính sách đẩy nhau, không cần chờ cái này hội tụ hẳn rồi mới chạy
cái kia. **Toàn bộ RL về sau là các biến thể của GPI**, chỉ khác cách ước lượng giá trị
(bằng mẫu thay vì bằng model).

Hạn chế: DP cần model đầy đủ và duyệt hết mọi trạng thái ⇒ "curse of dimensionality".

### Chương 5 — Monte Carlo Methods
Bỏ yêu cầu biết model: **chơi thật nhiều episode, lấy trung bình return** để ước lượng giá trị.

- **First-visit MC**: chỉ tính lần đầu tiên trạng thái xuất hiện trong episode.
- **Every-visit MC**: tính mọi lần xuất hiện.
- **MC control**: học `q_π` rồi cải thiện π. Cần đảm bảo khám phá: hoặc **exploring starts**,
  hoặc dùng chính sách **ε-soft**.
- **Off-policy + importance sampling**: học π (target) từ dữ liệu sinh bởi b (behavior).
  Trọng số `ρ = π(a|s)/b(a|s)` bù lại việc dữ liệu đến từ chính sách khác.
  - *Ordinary IS*: không chệch nhưng **phương sai vô hạn** (ví dụ 5.5 trong sách).
  - *Weighted IS*: chệch nhưng phương sai thấp, thực tế dùng tốt hơn nhiều.

Ưu: không chệch, không cần Markov hoàn hảo. Nhược: phải đợi hết episode, phương sai cao.

### Chương 6 — Temporal-Difference Learning
**Chương quan trọng nhất của cuốn sách.** TD học từng bước, không cần đợi hết episode,
bằng cách **bootstrap** — dùng ước lượng hiện tại để cập nhật ước lượng hiện tại:

```
TD(0):   V(S_t) ← V(S_t) + α [ R_{t+1} + γV(S_{t+1}) − V(S_t) ]
```

Đại lượng trong ngoặc là **TD error δ_t** — "phần thưởng cộng với dự đoán mới, trừ dự đoán cũ".
Đây chính là tín hiệu mà não (dopamine) dường như cũng dùng (Chương 15).

Ba thuật toán điều khiển:
- **Sarsa** (on-policy): cập nhật theo `(S,A,R,S',A')` — chính sách đang học cũng là chính
  sách đang hành động. "Học giá trị của chính mình".
- **Q-learning** (off-policy): `Q(S,A) ← Q + α[R + γ·max_{a'} Q(S',a') − Q]`. Học giá trị
  của chính sách **tối ưu** bất kể đang hành động ngu ngốc thế nào.
- **Expected Sarsa**: lấy kỳ vọng thay vì max ⇒ ít phương sai hơn, và là trường hợp riêng
  của Q-learning khi π tham lam.

Thêm **Double Learning** để chữa **maximization bias**: vì `max` của các ước lượng nhiễu
luôn thiên lên, nên tách thành hai bảng Q1, Q2 để chọn và đánh giá chéo nhau.

So sánh MC vs TD (mục 6.3): TD có phương sai thấp hơn, học online; MC không chệch nhưng
phương sai cao. TD(0) hội tụ về nghiệm của **bài toán dự đoán nhất quán** (certainty
equivalence) — tương đương mô hình Markov tối đa hoá hợp lý (maximum-likelihood MDP).

### Chương 7 — n-step Bootstrapping
Cầu nối giữa TD(0) (nhìn 1 bước) và MC (nhìn hết episode):

```
G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
```

`n=1` ⇒ TD(0), `n→∞` ⇒ MC. Thực nghiệm cho thấy `n` trung bình (4–10) thường tốt nhất.
Có các biến thể: n-step Sarsa, off-policy n-step Sarsa (importance sampling),
**Tree Backup** (không dùng IS, lan truyền ngược cây hành động), và **n-step Q(σ)** —
một thuật toán thống nhất, với σ=1 thành Sarsa, σ=0 thành Tree Backup.

### Chương 8 — Planning and Learning with Tabular Methods
Hợp nhất **learning** (học từ trải nghiệm thật) và **planning** (học từ trải nghiệm tưởng
tượng do model sinh ra). Cả hai đều là "cập nhật giá trị từ những gì đã biết".

- **Dyna-Q**: mỗi bước thật thì cập nhật Q, lưu model, rồi **mô phỏng thêm n bước** từ model.
  Chỉ thêm vài dòng code mà tốc độ học tăng gấp nhiều lần.
- **Prioritized sweeping**: không mô phỏng ngẫu nhiên, mà ưu tiên những `(s,a)` có TD error
  lớn (nơi "niềm tin đang sai nhiều nhất").
- **Expected vs sample updates**, **trajectory sampling** (chỉ lan truyền theo phân phối
  trạng thái thực tế gặp phải), **RTDP**, **rollout**, **MCTS** (dùng cho Go, cờ vây).

Cuối chương là mục "Summary of Part I: Dimensions" — bảng phân loại mọi thuật toán theo
các trục: độ sâu cập nhật (1 bước → n bước → λ), độ rộng (mẫu vs kỳ vọng), on/off-policy,
giá trị vs chính sách.

### Chương 9 — On-policy Prediction with Approximation
Bước ngoặt sang bài toán lớn: **không thể lưu bảng**. Thay `V(s)` bằng `v̂(s,w)` — hàm
tham số hoá (tuyến tính `wᵀx(s)`, hoặc neural network).

- Mục tiêu mới: **VE (Mean Squared Value Error)**, có trọng số là phân phối trạng thái `μ(s)`.
- **Semi-gradient**: vì target `R + γv̂(S',w)` cũng phụ thuộc `w`, nên không phải gradient
  thật; ta bỏ qua đạo hàm của target ⇒ "nửa gradient". Đây là mẹo kỹ thuật cốt lõi.
- **Linear methods**: `v̂ = wᵀx(s)`, cập nhật `w ← w + αδ_t x(s)`, hội tụ nếu α giảm đúng cách.
- **Feature construction**: polynomial, **Fourier basis**, **coarse coding**, **tile coding**
  (hiệu quả nhất trong thực hành cổ điển), RBF.
- **LSTD**: giải trực tiếp phương trình chuẩn, `O(d²)` trên mỗi bước, nhưng mẫu hiệu quả hơn.

### Chương 10 — On-policy Control with Approximation
Áp dụng xấp xỉ hàm cho điều khiển: `q̂(s,a,w)`, cập nhật semi-gradient với TD error.
Gồm **episodic semi-gradient Sarsa**, **n-step**, và bước chuyển sang bài toán **liên tục
không chiết khấu (average reward)**: thay `γ` bằng `r̄` (phần thưởng trung bình) và
dùng **differential return**. Thuật toán: differential semi-gradient Sarsa.
Sách cũng tranh luận mạnh: trong bài toán liên tục, **bỏ chiết khấu** hợp lý hơn về mặt
lý thuyết (mục 10.4, "Deprecating the Discounted Setting").

### Chương 11 — Off-policy Methods with Approximation
Chương "đen tối" nhất, và cũng là chương đáng đọc nhất với người làm deep RL.

- Off-policy + xấp xỉ hàm có thể **phân kỳ** (Baird's counterexample).
- **Deadly triad (bộ ba chết chóc)**: `xấp xỉ hàm` + `bootstrapping` + `off-policy`
  ⇒ mất ổn định. Chỉ cần bỏ một yếu tố là yên.
- Giải pháp: **Gradient-TD**, **TDC (GTD-2)**, **Emphatic-TD** — các phương pháp
  có bảo đảm hội tụ nhưng chi phí tính toán cao hơn.

Đây chính là nền tảng lý thuyết giải thích vì sao DQN cần replay buffer cố định, target
network, và vì sao huấn luyện deep RL hay "nổ".

### Chương 12 — Eligibility Traces
Cách thực hiện λ-return **online, mỗi bước một cập nhật, bộ nhớ O(d)** thay vì O(n).

```
z_t = γλ·z_{t-1} + ∇v̂(S_t,w)          (eligibility trace)
w   ← w + α·δ_t·z_t
```

`λ=0` ⇒ TD(0); `λ=1` ⇒ gần Monte Carlo. Đây là "cầu trượt" liên tục giữa hai cực.
Chương này khó nhất về kỹ thuật: **online λ-return**, **true online TD(λ)** (chính xác
từng bước), **Dutch traces**, **Sarsa(λ)**, **Watkins' Q(λ)**, **Tree-Backup(λ)**,
và các phương pháp off-policy ổn định. Nếu chỉ đọc một lần, hãy nắm ý tưởng: *trace là
bộ nhớ ngắn hạn ghi "trạng thái nào vừa góp phần gây ra sai số"*.

### Chương 13 — Policy Gradient Methods
Đổi cách tiếp cận: **học trực tiếp tham số của chính sách** `π(a|s,θ)` thay vì suy ra
chính sách từ giá trị.

**Định lý policy gradient:**
```
∇J(θ) ∝ E[ Σ_a q_π(S,a) · ∇π(a|S,θ) ] = E[ G_t · ∇ln π(A_t|S_t,θ) ]
```

- **REINFORCE**: dùng return `G_t` làm ước lượng của `q_π` ⇒ không chệch, phương sai cao.
- **REINFORCE with baseline**: trừ `b(s)` (thường là `v̂(s,w)`) ⇒ giảm phương sai mà không
  chệch, vì `Σ_a b(s)∇π(a|s) = 0`.
- **Actor–Critic**: actor là `π(a|s,θ)`, critic là `v̂(s,w)`; critic cung cấp TD error
  làm tín hiệu học cho actor, cập nhật **mỗi bước** thay vì cuối episode.
- Mở rộng: bài toán liên tục (differential), hành động liên tục (chính sách Gaussian
  với `μ(s,θ)`, `σ(s,θ)`).

Đây là nền của A3C, PPO, SAC, và cả **RLHF** dùng để huấn luyện ChatGPT.

### Chương 14 — Psychology
So sánh TD với các mô hình học tập của động vật: điều kiện hoá cổ điển (Pavlov),
điều kiện hoá công cụ (Thorndike), mô hình **Rescorla–Wagner**, và mô hình TD
(Sutton & Barto 1981). Điểm thú vị: mô hình TD giải thích được hiệu ứng **blocking**
và **higher-order conditioning** — những hiện tượng mà Rescorla–Wagner không giải thích nổi.
Cũng bàn về **cognitive maps** (bản đồ nhận thức ở chuột) và sự khác nhau giữa
hành vi **habitual** (thói quen, model-free) và **goal-directed** (có chủ đích, model-based).

### Chương 15 — Neuroscience
Câu chuyện nổi tiếng nhất của RL trong sinh học: **dopamine = TD error**.
Các neuron dopamine ở vùng VTA/substantia nigra phát xung đúng theo mẫu:
- Phần thưởng bất ngờ ⇒ tăng vọt (positive prediction error)
- Phần thưởng đã dự đoán trước ⇒ không phản ứng
- Mất phần thưởng đã dự đoán ⇒ giảm dưới baseline (negative prediction error)

Sách trình bày mô hình **actor–critic trong hạch nền (basal ganglia)**, các thí nghiệm
kinh điển (Schultz, Montague, Dayan), và các chủ đề mở rộng: hedonistic neurons,
học tập tập thể, model-based ở não, và **nghiện** như một dạng RL bị lệch.

### Chương 16 — Applications and Case Studies
Các case study lịch sử, đọc rất "đã":
- **TD-Gammon** (Tesauro 1992): chơi backgammon ngang cao thủ nhờ TD(λ) + mạng nơ-ron.
  Đây là bằng chứng sớm nhất rằng RL + xấp xỉ hàm có thể mạnh.
- **Samuel's checkers** (1959) — tổ tiên của mọi thứ.
- **Watson's Daily-Double wagering** (IBM) — RL cho việc đặt cược.
- **Tối ưu điều khiển bộ nhớ** trong hệ điều hành.
- **DQN chơi Atari** (Mnih et al. 2015) — deep RL bùng nổ.
- **AlphaGo / AlphaGo Zero** (Silver et al.) — MCTS + deep network + self-play.
- **Dịch vụ web cá nhân hoá**, **thermal soaring** (dùng RL điều khiển tàu lượn lợi dụng
  cột khí nóng — ví dụ đẹp nhất về RL ngoài đời thật).

### Chương 17 — Frontiers
Các hướng mở: **General Value Functions** và auxiliary tasks, **options** (trừu tượng hoá
thời gian — học "làm sao để làm một việc nhỏ" rồi ghép lại), vấn đề **state/observation**
(POMDP), **thiết kế phần thưởng** (reward hacking, an toàn), và các câu hỏi còn bỏ ngỏ:
credit assignment, khám phá trong không gian lớn, học hiệu quả mẫu, và vai trò của RL
trong con đường tới AGI.

---

## 4. Hệ thống thuật toán của cuốn sách

### 4.1. Bảng phân loại

| Thuật toán | Chương | Cần model? | On/Off-policy | Bootstrapping | Ghi chú |
|---|---|---|---|---|---|
| ε-greedy | 2 | – | – | – | baseline bandit |
| UCB | 2 | – | – | – | khám phá có định hướng |
| Gradient bandit | 2 | – | – | – | tiền thân policy gradient |
| Policy Evaluation | 4 | ✅ | on | ✅ | 1 sweep = 1 bước |
| Policy Iteration | 4 | ✅ | on | ✅ | eval ↔ improve |
| Value Iteration | 4 | ✅ | on | ✅ | 1 sweep gộp |
| First-visit MC | 5 | ❌ | on | ❌ | chờ hết episode |
| Monte Carlo ES | 5 | ❌ | on | ❌ | exploring starts |
| Off-policy MC | 5 | ❌ | off | ❌ | importance sampling |
| TD(0) | 6 | ❌ | on | ✅ | 1 bước |
| Sarsa | 6 | ❌ | on | ✅ | học giá trị của π |
| Q-learning | 6 | ❌ | off | ✅ | học π* |
| Expected Sarsa | 6 | ❌ | on/off | ✅ | ít phương sai hơn |
| Double Q-learning | 6 | ❌ | off | ✅ | chữa maximization bias |
| n-step TD / Sarsa | 7 | ❌ | on | ✅ | trung gian TD↔MC |
| Tree Backup / Q(σ) | 7 | ❌ | off | ✅ | không cần IS |
| Dyna-Q | 8 | model học được | off | ✅ | học + planning |
| Prioritized Sweeping | 8 | model học được | off | ✅ | ưu tiên TD error lớn |
| Semi-gradient TD | 9 | ❌ | on | ✅ | xấp xỉ hàm |
| LSTD | 9 | ❌ | on | ✅ | O(d²)/bước |
| Semi-gradient Sarsa | 10 | ❌ | on | ✅ | điều khiển + FA |
| Differential Sarsa | 10 | ❌ | on | ✅ | average reward |
| Gradient-TD / TDC | 11 | ❌ | off | ✅ | hội tụ dưới deadly triad |
| TD(λ), Sarsa(λ) | 12 | ❌ | on | ✅ | trace |
| True Online TD(λ) | 12 | ❌ | on | ✅ | chính xác từng bước |
| REINFORCE | 13 | ❌ | on | ❌ | policy gradient |
| Actor–Critic | 13 | ❌ | on | ✅ | actor + critic |

### 4.2. Ba trục phân loại quan trọng nhất

1. **Có model hay không (model-based vs model-free).** Có model ⇒ planning, mẫu hiệu quả
   hơn, nhưng model sai thì học sai (mục 8.3).
2. **On-policy vs off-policy.** On-policy học giá trị của chính chính sách đang chạy
   (an toàn, ổn định). Off-policy học giá trị của chính sách *mục tiêu*, tận dụng được
   dữ liệu cũ, nhưng khó hội tụ hơn nhiều.
3. **Độ sâu bootstrapping (1 bước → n bước → λ).** Càng bootstrap nhiều ⇒ phương sai thấp,
   chệch cao; càng gần MC ⇒ không chệch, phương sai cao. Chương 7 và 12 chính là hai cách
   di chuyển trên trục này.

---

## 5. Những ý tưởng lớn cần nhớ

1. **Giá trị là trung tâm.** Reward là tín hiệu, nhưng *value* là thứ được học. Mọi thuật
   toán đều quy về việc làm cho ước lượng `v̂` tiến gần `v_π` hoặc `v*`.
2. **Bootstrapping là con dao hai lưỡi.** Nó cho phép học online, hiệu quả mẫu cao, nhưng
   tạo ra bias và — khi ghép với xấp xỉ hàm + off-policy — gây phân kỳ.
3. **Exploration không thể bỏ qua.** Không có khám phá thì không có tối ưu. Sách dành
   hẳn chương 2 và rất nhiều mục ở chương 8, 17 để nói về nó, và thẳng thắn thừa nhận
   đây là vấn đề **chưa được giải quyết trọn vẹn**.
4. **Không có thuật toán vạn năng.** Mỗi phương pháp đánh đổi bias–variance, bộ nhớ,
   tính toán, và yêu cầu về dữ liệu khác nhau. Chọn thuật toán là chọn **đánh đổi phù hợp
   với bài toán**, không phải tìm "cái tốt nhất".
5. **GPI là khung chung.** Hầu hết thuật toán = đánh giá chính sách + cải thiện chính sách,
   lặp qua lặp lại, chỉ khác cách ước lượng.

---

## 6. Đánh giá phê bình

**Điểm mạnh**
- Là cuốn sách *duy nhất* trình bày RL thành một hệ thống lý thuyết mạch lạc, từ bandit
  đến policy gradient, không bỏ qua nền tảng toán.
- Giải thích **tại sao**, không chỉ **như thế nào**. Ví dụ: mục 6.3 giải thích TD(0) hội tụ
  về nghiệm maximum-likelihood; mục 11.2 chỉ ra chính xác cơ chế phân kỳ.
- Hệ thống bài tập và hình minh hoạ cực tốt (hàng trăm hình, mỗi hình là một thí nghiệm
  tái lập được).
- Cập nhật kịp thời đại: có DQN, AlphaGo, RLHF-adjacent.

**Điểm yếu / lưu ý khi đọc**
- **Không dạy lập trình.** Sách không có code; người đọc phải tự cài đặt (đó là lý do
  tôi viết kèm bộ file `algorithms/`).
- Phần II (chương 9–13) rất nặng về toán; người mới nên đọc chậm hoặc bỏ qua các mục
  có dấu `*`.
- Xuất bản 2018 nên **không có** PPO, SAC, TD3, Dreamer, RLHF chi tiết — những thứ
  thống trị thực tế hiện nay. Tuy vậy, nền tảng trong sách là điều kiện cần để hiểu chúng.
- Giả định MDP đầy đủ quan sát; POMDP chỉ được nhắc ở chương 17.

**Nên đọc thế nào cho hiệu quả**
- Lần 1: Chương 1, 3, 6 (bỏ qua công thức khó), 13. Nắm vòng lặp agent–môi trường,
  TD error, policy gradient.
- Lần 2: Chương 2, 4, 5 để hiểu tại sao cần TD.
- Lần 3: Chương 7, 8, 12 (n-step, Dyna, trace) — đây là phần "gia vị" giúp thuật toán
  chạy tốt hơn trong thực tế.
- Chương 9–11 chỉ đọc khi đã làm việc với deep RL.

---

## 7. Kết luận

*Reinforcement Learning: An Introduction* không phải sách "đọc xong để biết", mà là sách
**để tra cứu và để làm**. Giá trị lớn nhất của nó không nằm ở các thuật toán cụ thể
(nhiều cái đã lỗi thời về hiệu năng), mà ở **cách tư duy**: biến bài toán ra quyết định
thành bài toán ước lượng giá trị, rồi cải thiện chính sách dựa trên ước lượng đó.
Chỉ cần nắm chắc ba khái niệm — **value, bootstrapping, exploration** — là có thể đọc
hiểu gần như mọi bài báo RL hiện đại.

---

## 8. Cấu trúc bộ file kèm theo

Báo cáo này đi kèm thư mục `algorithms/`, **mỗi file là một thuật toán**, code Python
chạy được ngay, có:

- Docstring tiếng Việt: bài toán → ý tưởng → công thức → pseudocode của sách → cách đọc code.
- Code có đánh số bước `(1)`, `(2)`… khớp với pseudocode trong sách.
- Phần `__main__`: chạy thử trên bài toán đồ chơi nhỏ, in kết quả ra màn hình.

| File | Thuật toán | Chương |
|---|---|---|
| `01_epsilon_greedy.py` | ε-greedy + cập nhật trung bình tăng dần | 2.3–2.5 |
| `02_ucb.py` | Upper-Confidence-Bound | 2.7 |
| `03_gradient_bandit.py` | Gradient bandit (softmax) | 2.8 |
| `04_policy_evaluation.py` | Iterative Policy Evaluation | 4.1 |
| `05_policy_iteration.py` | Policy Iteration | 4.3 |
| `06_value_iteration.py` | Value Iteration | 4.4 |
| `07_monte_carlo_prediction.py` | First-visit / Every-visit MC | 5.1 |
| `08_monte_carlo_control.py` | Monte Carlo ES | 5.3–5.4 |
| `09_off_policy_mc.py` | Off-policy MC + Importance Sampling | 5.5–5.7 |
| `10_td0.py` | Tabular TD(0) | 6.1 |
| `11_sarsa.py` | Sarsa (on-policy TD control) | 6.4 |
| `12_q_learning.py` | Q-learning (off-policy TD control) | 6.5 |
| `13_expected_sarsa.py` | Expected Sarsa | 6.6 |
| `14_double_q_learning.py` | Double Q-learning | 6.7 |
| `15_n_step_td.py` | n-step TD | 7.1 |
| `16_n_step_sarsa.py` | n-step Sarsa | 7.2 |
| `17_off_policy_n_step_sarsa.py` | Off-policy n-step Sarsa | 7.3 |
| `18_tree_backup.py` | n-step Tree Backup | 7.5 |
| `19_n_step_q_sigma.py` | n-step Q(σ) | 7.6 |
| `20_dyna_q.py` | Dyna-Q | 8.2 |
| `21_prioritized_sweeping.py` | Prioritized Sweeping | 8.4 |
| `22_semi_gradient_td.py` | Semi-gradient TD(0) | 9.3 |
| `23_semi_gradient_sarsa.py` | Episodic semi-gradient Sarsa | 10.1 |
| `24_n_step_semi_gradient_sarsa.py` | n-step semi-gradient Sarsa | 10.2 |
| `25_lstd.py` | Least-Squares TD | 9.8 |
| `26_differential_sarsa.py` | Differential semi-gradient Sarsa | 10.5 |
| `27_td_lambda.py` | TD(λ) với eligibility trace | 12.2–12.3 |
| `28_true_online_td_lambda.py` | True online TD(λ) | 12.5 |
| `29_sarsa_lambda.py` | Sarsa(λ) | 12.7 |
| `30_true_online_sarsa_lambda.py` | True online Sarsa(λ) | 12.7 |
| `31_reinforce.py` | REINFORCE | 13.3 |
| `32_reinforce_baseline.py` | REINFORCE with baseline | 13.4 |
| `33_actor_critic.py` | One-step Actor–Critic + trace | 13.5 |

Cách chạy:

```bash
cd algorithms
python 12_q_learning.py
```

Yêu cầu: Python 3.10+ và `numpy`.

---

## 9. Tài liệu tham khảo

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
   (2nd ed.). MIT Press.
2. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
3. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards* (Luận án tiến sĩ, Cambridge).
4. Tesauro, G. (1995). Temporal difference learning and TD-Gammon.
   *Communications of the ACM*, 38(3), 58–68.
5. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning.
   *Nature*, 518, 529–533.
6. Silver, D. et al. (2016). Mastering the game of Go with deep neural networks
   and tree search. *Nature*, 529, 484–489.
7. Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction
   and reward. *Science*, 275, 1593–1599.
