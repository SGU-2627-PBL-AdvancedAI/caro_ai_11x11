# CARO AI 11×11
## Tài liệu dự án, Notebook thực nghiệm và các tác tử Random / Rule-Based

> Tài liệu này được viết dựa trên mã nguồn trong gói `caro_ai_11x11` được cung cấp. Nội dung phân biệt rõ những gì đã có trong mã nguồn và những phần mới chỉ là khung/dự kiến.

---

## 1. Tổng quan dự án

Dự án xây dựng môi trường Caro 11×11 và các tác tử AI có thể chơi với nhau hoặc chơi với người dùng. Notebook hiện có tập trung vào hai baseline:

- **Random Agent**: chọn ngẫu nhiên một nước đi hợp lệ.
- **Rule-Based Agent**: sử dụng heuristic theo các cửa sổ 5 ô để ưu tiên thắng ngay, chặn đối thủ và tạo thế tấn công/phòng thủ.

Notebook chính:

```text
notebooks/01_rulebase_random.ipynb
```

Các thành phần cốt lõi:

```text
src/caro_env.py
src/agents/base_agent.py
src/agents/random_agent.py
src/agents/rule_based_agent.py
src/tournament.py
```

Notebook thực hiện kiểm tra môi trường, smoke test hai agent, chạy một ván minh họa, tournament, tổng hợp CSV, vẽ biểu đồ, xuất video MP4 và có phần giao diện Human vs AI bằng `ipywidgets`.

### 1.1. Quy tắc trò chơi

| Thuộc tính | Giá trị |
|---|---|
| Kích thước bàn cờ mặc định | 11 × 11 |
| Số ô | 121 |
| Độ dài chuỗi thắng | 5 quân liên tiếp |
| Người chơi X | `1`, đi trước mặc định |
| Người chơi O | `-1`, đi sau mặc định |
| Ô trống | `0` |
| Hành động | Số nguyên từ `0` đến `120` |
| Cách thắng | Có ít nhất 5 quân liên tiếp theo hàng, cột hoặc một trong hai đường chéo |

Một nước đi tại hàng `r`, cột `c` được mã hóa thành:

```python
action = r * board_size + c
```

Giải mã:

```python
r, c = divmod(action, board_size)
```

Ví dụ trên bàn 11×11, ô hàng 2 cột 3 có action `2 * 11 + 3 = 25`.

---

## 2. Cấu trúc thư mục

Cấu trúc rút gọn của gói dự án:

```text
caro_ai_11x11/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── default.yaml
├── data/
├── checkpoints/
├── docs/
│   ├── chuong_1/
│   ├── chuong_2/
│   ├── chuong_3/
│   ├── chuong_4/
│   └── chuong_5/
├── notebooks/
│   └── 01_rulebase_random.ipynb
├── results/
│   ├── charts/
│   ├── tables/
│   └── videos/
├── src/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── random_agent.py
│   │   ├── rule_based_agent.py
│   │   ├── minimax_agent.py
│   │   ├── mcts_agent.py
│   │   └── cnn_agent.py
│   ├── caro_env.py
│   ├── tournament.py
│   └── ui_app.py
├── tests/
│   └── test_caro_env.py
├── main.py
├── requirements.txt
├── pytest.ini
└── README.md
```

### 2.1. Vai trò từng thành phần

| Đường dẫn | Vai trò |
|---|---|
| `src/caro_env.py` | Luật chơi, trạng thái bàn cờ, nước đi hợp lệ, kiểm tra thắng/thua/hòa, undo và clone |
| `src/agents/base_agent.py` | Lớp cơ sở quy định giao diện chung cho các agent |
| `src/agents/random_agent.py` | Agent chọn ngẫu nhiên trong các nước đi hợp lệ |
| `src/agents/rule_based_agent.py` | Agent heuristic dựa trên cửa sổ 5 ô |
| `src/agents/minimax_agent.py` | Có file mã nguồn trong gói; cần kiểm tra/kiểm thử riêng trước khi xem là baseline đã xác nhận |
| `src/agents/mcts_agent.py` | Có file mã nguồn trong gói; cần kiểm tra/kiểm thử riêng |
| `src/agents/cnn_agent.py` | Có file mã nguồn trong gói; cần kiểm tra/kiểm thử riêng |
| `src/tournament.py` | Chạy một ván và nhiều ván đối kháng, đo thời gian chọn nước |
| `notebooks/01_rulebase_random.ipynb` | Notebook thực nghiệm Random và Rule-Based, thống kê, biểu đồ, video, Human vs AI |
| `results/tables/` | CSV kết quả |
| `results/charts/` | Biểu đồ PNG |
| `results/videos/` | Video MP4 |
| `configs/default.yaml` | Cấu hình mặc định cho trò chơi và thực nghiệm |
| `tests/` | Kiểm thử tự động |
| `main.py` | Hiện đang để trống trong gói được cung cấp |
| `src/ui_app.py` | Hiện đang để trống trong gói được cung cấp |

Các thư mục `data/`, `checkpoints/` và `docs/chuong_1..chuong_5/` hiện là vị trí tổ chức dữ liệu/tài liệu; không nên hiểu rằng dự án đã có dữ liệu huấn luyện hoặc checkpoint mô hình chỉ vì các thư mục này tồn tại.

---

## 3. Mô hình trạng thái và API của CaroEnv

Lớp `CaroEnv` trong `src/caro_env.py` quản lý toàn bộ trạng thái một ván.

### 3.1. Biểu diễn bàn cờ

Bàn cờ là mảng NumPy kiểu `int8`:

```python
env.board.shape  # (11, 11)
```

Quy ước giá trị:

```text
 1  = X
-1  = O
 0  = ô trống
```

Các thuộc tính trạng thái quan trọng:

| Thuộc tính | Ý nghĩa |
|---|---|
| `board` | Ma trận bàn cờ |
| `board_size` | Kích thước cạnh bàn |
| `win_length` | Số quân liên tiếp cần để thắng |
| `current_player` | Người chơi đang tới lượt (`1` hoặc `-1`) |
| `history` | Danh sách action đã thực hiện |
| `_done` | Cờ nội bộ cho biết ván đã kết thúc |
| `winner` | `1` nếu X thắng, `-1` nếu O thắng, `0` nếu hòa/chưa có người thắng |
| `last_player` | Người chơi thực hiện nước đi gần nhất |

Lưu ý: `CaroEnv` sử dụng `_done` nội bộ và cung cấp `is_terminal()`; không có thuộc tính `env.done` trong mã nguồn hiện tại. Số nước đã đi được lấy bằng `len(env.history)`; không có thuộc tính `env.move_count`.

### 3.2. Các phương thức chính

#### `reset()`

Tạo bàn cờ rỗng và khôi phục trạng thái đầu ván:

```python
env.reset()
```

Hành vi trong mã nguồn:

- Tạo ma trận toàn số 0.
- Đặt `current_player = 1`.
- Xóa `history`.
- Đặt `_done = False`.
- Đặt `winner = 0`.
- Đặt `last_player = 0`.
- Trả về bản sao của bàn cờ.

#### `legal_actions()`

Trả về danh sách action của các ô còn trống:

```python
legal = env.legal_actions()
```

Đây là **method**, vì vậy phải gọi dấu ngoặc `()`. Khi ván đã kết thúc, phương thức trả về danh sách rỗng.

#### `step(action)`

Thực hiện một nước đi hợp lệ:

```python
board, reward, done, info = env.step(action)
```

Giá trị trả về:

- `board`: bản sao bàn cờ sau nước đi.
- `reward`: `1.0` khi nước đi vừa tạo chiến thắng; `0.0` trong các trường hợp còn lại.
- `done`: ván đã kết thúc hay chưa.
- `info`: thông tin nước đi, người vừa đi, người thắng và số nước đi.

`step()` kiểm tra action có phải số nguyên, có nằm trong phạm vi bàn cờ và ô đích có đang trống hay không. Sau nước đi, môi trường kiểm tra chiến thắng; nếu chưa thắng và bàn đầy thì kết thúc hòa. Nếu ván chưa kết thúc, lượt chơi được đổi sang đối thủ.

#### `is_terminal()`

```python
if env.is_terminal():
    ...
```

Trả về trạng thái kết thúc của ván.

#### `_check_win_from_cell(r, c, player)`

Kiểm tra chuỗi quân thắng đi qua ô vừa đánh theo bốn hướng:

1. Ngang: `(0, 1)`.
2. Dọc: `(1, 0)`.
3. Chéo xuống: `(1, 1)`.
4. Chéo lên: `(1, -1)`.

Hàm đếm quân cùng màu ở cả hai phía của ô vừa đánh và so sánh với `win_length`.

Đây là phương thức nội bộ (có dấu gạch dưới đầu tên), được Rule-Based Agent sử dụng để nhận biết nước thắng ngay.

#### `clone()`

```python
simulated_env = env.clone()
```

Tạo môi trường bản sao độc lập: ma trận, lượt chơi, lịch sử, trạng thái kết thúc, người thắng và người đi gần nhất đều được sao chép. Thay đổi bản sao không làm thay đổi môi trường gốc.

#### `undo()`

Hoàn tác nước đi gần nhất, xóa quân ở ô tương ứng và khôi phục lượt chơi cũng như các trường trạng thái kết thúc.

#### `result(perspective=1)`

Trả về kết quả theo góc nhìn của người chơi:

- `1.0`: người chơi được chỉ định thắng.
- `-1.0`: người chơi được chỉ định thua.
- `0.0`: hòa hoặc ván chưa kết thúc.

---

## 4. Giao diện chung của Agent

`src/agents/base_agent.py` định nghĩa lớp trừu tượng `BaseAgent`.

```python
class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(self, env) -> int:
        raise NotImplementedError
```

Mỗi agent cần:

- Có tên `name`.
- Cài đặt `select_action(env)`.
- Nhận môi trường Caro hiện tại.
- Trả về một action hợp lệ trong danh sách `env.legal_actions()`.

Quy ước đầu ra là action dạng số nguyên `0..120` với bàn 11×11. Agent không nên tự ý thực hiện nước đi bằng `env.step()` trong `select_action()`; nhiệm vụ của agent là chọn action, còn môi trường/tournament chịu trách nhiệm thực hiện action.

---

## 5. Random Agent

Tệp: `src/agents/random_agent.py`

### 5.1. Mục tiêu

Random Agent là baseline đơn giản nhất. Agent không đánh giá chiến thuật, không dự đoán nước đi tiếp theo và không ưu tiên tấn công hay phòng thủ. Nó chọn ngẫu nhiên một phần tử trong tập nước đi hợp lệ.

### 5.2. Khởi tạo

```python
from src.agents.random_agent import RandomAgent

agent = RandomAgent(seed=42, name="Random")
```

- `seed`: seed cho bộ sinh số ngẫu nhiên riêng `random.Random`.
- `name`: tên agent dùng trong log và báo cáo.
- Nếu không truyền seed, kết quả lựa chọn không được cố định theo seed do người dùng chỉ định.

### 5.3. Thuật toán chọn nước đi

Logic trong mã nguồn:

```python
def select_action(self, env):
    legal = env.legal_actions()

    if not legal:
        raise RuntimeError("Không còn nước đi hợp lệ.")

    return self.rng.choice(legal)
```

Quy trình:

1. Gọi `env.legal_actions()` để lấy các ô còn trống.
2. Nếu danh sách rỗng, phát sinh `RuntimeError`.
3. Dùng bộ sinh số ngẫu nhiên riêng của agent chọn một action trong danh sách.
4. Trả action về cho nơi gọi.

### 5.4. Đặc điểm

**Ưu điểm**

- Cài đặt đơn giản, dễ kiểm tra.
- Chi phí tính toán rất thấp.
- Phù hợp làm baseline ngẫu nhiên để so sánh với heuristic hoặc thuật toán tìm kiếm.
- Có thể tái lập chuỗi lựa chọn khi sử dụng seed cố định và giữ nguyên thứ tự lời gọi.

**Hạn chế**

- Không nhận biết nước thắng ngay.
- Không chặn nước thắng của đối thủ.
- Không đánh giá cấu trúc chuỗi quân.
- Không có chiến lược dài hạn.
- Chất lượng chơi không ổn định và phụ thuộc nhiều vào yếu tố ngẫu nhiên.

### 5.5. Ví dụ kiểm tra

```python
from src.caro_env import CaroEnv
from src.agents.random_agent import RandomAgent

env = CaroEnv(board_size=11, win_length=5)
agent = RandomAgent(seed=42)

action = agent.select_action(env)

assert action in env.legal_actions()

row, col = divmod(action, env.board_size)
print("Action:", action)
print("Vị trí:", row, col)
```

---

## 6. Rule-Based Agent

Tệp: `src/agents/rule_based_agent.py`

### 6.1. Mục tiêu

Rule-Based Agent là baseline heuristic có thể giải thích. Thay vì chọn ngẫu nhiên, agent đánh giá các nước đi dựa trên những đoạn liên tiếp dài 5 ô.

Chiến lược tổng quát theo docstring và mã nguồn:

1. Ưu tiên nước đi thắng ngay.
2. Ưu tiên ngăn đối thủ thắng ngay.
3. Tạo hoặc mở rộng các cấu trúc có lợi cho bản thân.
4. Cân bằng điểm tấn công và phòng thủ.
5. Ưu tiên khu vực gần trung tâm khi điểm chiến thuật tương đương.

Đây là heuristic một bước, không phải Minimax, Alpha-Beta hay MCTS.

### 6.2. Trọng số heuristic

Trong constructor:

```python
self.weights = {
    5: 10**9,
    4: 200000,
    3: 12000,
    2: 700,
    1: 30,
    0: 0
}
```

Ý nghĩa:

| Số quân cùng phe trong một cửa sổ 5 ô | Trọng số cơ sở |
|---:|---:|
| 5 | 1,000,000,000 |
| 4 | 200,000 |
| 3 | 12,000 |
| 2 | 700 |
| 1 | 30 |
| 0 | 0 |

Trọng số tăng mạnh theo số quân cùng phe để ưu tiên những cửa sổ có tiềm năng tạo chuỗi dài. Trường hợp có cả quân của hai bên trong cùng cửa sổ sẽ bị bỏ qua trong `_score_move()`.

Lưu ý: trọng số là giá trị heuristic do chương trình định nghĩa, không phải xác suất thắng.

### 6.3. Hàm `_score_move(board, r, c, player)`

Hàm đánh giá một vị trí giả định `(r, c)` cho `player`.

Với mỗi nước đi, hàm xét bốn hướng:

```python
((0, 1), (1, 0), (1, 1), (1, -1))
```

Trong mỗi hướng, hàm duyệt tối đa 5 cửa sổ có chứa vị trí đang đánh. Mỗi cửa sổ gồm đúng 5 ô.

Với từng cửa sổ:

- `own`: số ô có quân của người chơi đang được đánh giá.
- `opp`: số ô có quân đối thủ.
- Nếu `own > 0` và `opp > 0`: bỏ qua cửa sổ vì đã bị chặn bởi cả hai phe.
- Nếu chỉ có quân của người chơi: cộng `weights[own]`.
- Nếu chỉ có quân đối thủ: cộng `weights[opp] * 1.08`.

Hệ số `1.08` làm điểm phòng thủ cao hơn 8% so với trọng số tương ứng, nhằm ưu tiên chặn mối đe dọa.

Sau khi cộng điểm các cửa sổ, hàm cộng điểm ưu tiên trung tâm:

```python
center = (n - 1) / 2
total_score += max(
    0,
    10 - (abs(r-center) + abs(c-center))
)
```

Đây là điểm phụ theo khoảng cách Manhattan tới trung tâm. Điểm trung tâm chỉ có tác dụng phụ khi so sánh các nước đi có điểm chiến thuật gần nhau; nó không thay thế điểm cửa sổ 5 ô.

### 6.4. Giới hạn tập ứng viên

Trong `select_action()`:

- Agent lấy danh sách nước đi hợp lệ.
- Tìm các ô đã có quân bằng `np.argwhere(board != 0)`.
- Nếu bàn đã có quân, chỉ giữ các action có khoảng cách Chebyshev tới ít nhất một quân hiện có không quá 4:

```python
max(abs(r0-r), abs(c0-c)) <= 4
```

Mục tiêu là giảm số nước phải chấm điểm vì một nước đi quá xa mọi quân hiện có khó tạo thành chuỗi 5 ngay ở lượt hiện tại.

- Nếu danh sách ứng viên sau lọc rỗng, agent quay lại dùng toàn bộ nước đi hợp lệ.
- Nếu bàn cờ trống, agent chỉ xét ô trung tâm.

Với bàn 11×11, ô trung tâm là `(5, 5)`, tương ứng action `60`.

### 6.5. Chấm điểm tấn công và phòng thủ

Với mỗi action ứng viên:

1. Giải mã action thành hàng và cột.
2. Đặt tạm quân của người chơi hiện tại vào ô đó.
3. Gọi `_check_win_from_cell()` để kiểm tra có thắng ngay không.
4. Nếu thắng, hoàn tác ô giả lập và trả về action ngay.
5. Nếu chưa thắng, tính điểm tấn công:

```python
attack = self._score_move(board, r, c, player)
```

6. Đổi tạm quân tại ô đó thành quân đối thủ để mô phỏng việc đối thủ chiếm vị trí này.
7. Tính điểm phòng thủ:

```python
defense = self._score_move(board, r, c, -player)
```

8. Khôi phục ô về `0`.
9. Tính điểm tổng:

```python
score = attack + 0.98 * defense
```

10. Chọn action có điểm cao nhất.

### 6.6. Công thức điểm tổng

Có thể mô tả logic đánh giá bằng:

\[
Score(a) = Attack(a) + 0.98 \times Defense(a)
\]

Trong đó:

- `Attack(a)`: tổng điểm các cửa sổ 5 ô sau khi giả lập người chơi hiện tại chiếm action `a`.
- `Defense(a)`: tổng điểm các cửa sổ 5 ô khi giả lập đối thủ chiếm vị trí `a`.
- `0.98`: hệ số cân bằng được mã nguồn sử dụng.

Nước thắng ngay được xử lý riêng và trả về trước khi so sánh điểm tổng.

### 6.7. Ưu điểm và hạn chế

**Ưu điểm**

- Có chiến thuật rõ ràng hơn Random Agent.
- Có thể giải thích tại sao một nước đi được ưu tiên qua điểm cửa sổ 5 ô.
- Có kiểm tra thắng ngay.
- Có thành phần phòng thủ để chặn các đường phát triển nguy hiểm.
- Giảm số ứng viên cần đánh giá bằng vùng lân cận các quân đã có.
- Không cần dữ liệu huấn luyện hoặc GPU.

**Hạn chế**

- Chỉ đánh giá nước đi hiện tại và các cửa sổ cục bộ; không tìm kiếm nhiều ply.
- Không mô phỏng đầy đủ chuỗi phản ứng qua nhiều lượt.
- Không dùng Minimax/Alpha-Beta, MCTS hoặc mạng nơ-ron.
- Heuristic phụ thuộc vào bộ trọng số cố định.
- Không có đánh giá riêng tường minh cho các dạng mở/đóng hai đầu; trạng thái được phản ánh gián tiếp qua các cửa sổ 5 ô.
- Đây là baseline có thể giải thích, không nên mô tả là AI tối ưu hay có khả năng dự đoán chiến thuật dài hạn.

### 6.8. Ví dụ kiểm tra

```python
from src.caro_env import CaroEnv
from src.agents.rule_based_agent import RuleBasedAgent

env = CaroEnv(board_size=11, win_length=5)
agent = RuleBasedAgent()

action = agent.select_action(env)

assert action in env.legal_actions()

row, col = divmod(action, env.board_size)
print("Action:", action)
print("Vị trí:", row, col)
```

---

## 7. So sánh Random và Rule-Based

| Tiêu chí | Random Agent | Rule-Based Agent |
|---|---|---|
| Cách chọn nước | Chọn ngẫu nhiên | Chấm điểm heuristic |
| Kiểm tra nước hợp lệ | Có | Có |
| Ưu tiên thắng ngay | Không | Có |
| Chặn nước thắng đối thủ | Không | Có qua điểm phòng thủ |
| Đánh giá cửa sổ 5 ô | Không | Có |
| Ưu tiên trung tâm | Không | Có |
| Seed | Có | Không cần seed trong implementation hiện tại |
| Chi phí tính toán | Rất thấp | Cao hơn Random nhưng vẫn là heuristic cục bộ |
| Tìm kiếm nhiều lượt | Không | Không |
| Học từ dữ liệu | Không | Không |

Không nên kết luận Rule-Based luôn thắng Random trong mọi ván. Kết quả thực nghiệm cần dựa trên tournament đủ số ván, cân bằng vai trò X/O và ghi nhận seed/cấu hình.

---

## 8. Notebook `01_rulebase_random.ipynb`

### 8.1. Mục tiêu notebook

Notebook này là quy trình thực nghiệm đầu tiên của dự án. Nó dùng trực tiếp các module trong `src/`, tránh viết lại thuật toán agent trong notebook.

Notebook có các nhóm công việc:

1. Import thư viện, nạp module và cấu hình.
2. Kiểm tra môi trường Caro.
3. Smoke test hai agent.
4. Chạy một ván minh họa.
5. Chạy tournament giữa bốn cấu hình đối kháng.
6. Tổng hợp win/draw rate và số nước đi.
7. Phân tích thời gian chọn action.
8. Vẽ biểu đồ.
9. Ghi video MP4.
10. Liệt kê các kết quả được sinh ra.
11. Chơi Human vs AI bằng `ipywidgets`.
12. Ghi nhận xét và giới hạn phương pháp.

### 8.2. Cell 1 — Import và cấu hình

Notebook:

- Xác định `PROJECT_ROOT` từ thư mục làm việc hiện tại.
- Thêm project root vào `sys.path`.
- Import `CaroEnv`, `RandomAgent`, `RuleBasedAgent`.
- Reload `src.tournament` để tránh sử dụng phiên bản module cũ đã nằm trong cache kernel.
- Kiểm tra hai API `play_game` và `run_matchup`.
- Tạo các thư mục kết quả nếu chưa tồn tại.

Cấu hình hiện tại trong notebook:

```python
BOARD_SIZE = 11
WIN_LENGTH = 5
GAMES_PER_MATCHUP = 20
SEED = 42
```

`GAMES_PER_MATCHUP = 20` phù hợp chạy thử nhanh. Notebook ghi chú có thể đổi thành `100` cho thực nghiệm chính thức. Nếu thay đổi số ván, cần chạy lại các cell tournament và thống kê để CSV/biểu đồ đồng bộ.

### 8.3. Cell 2 — Kiểm tra môi trường

Khởi tạo:

```python
env = CaroEnv(BOARD_SIZE, WIN_LENGTH)
```

In ra:

- Kích thước ma trận.
- Số nước đi hợp lệ ban đầu.
- Bàn cờ dạng text thông qua `env.render()`.

Mục đích là xác nhận môi trường tạo đúng bàn 11×11 và trạng thái ban đầu có các ô trống.

### 8.4. Cell 3 — Smoke test Agent

Tạo hai agent:

```python
agents_smoke_test = [
    RandomAgent(seed=SEED),
    RuleBasedAgent()
]
```

Với từng agent:

- Gọi `select_action(env)`.
- Lấy `env.legal_actions()`.
- In tên agent, action, hàng, cột và trạng thái hợp lệ.
- Dùng `assert` để bảo đảm action được trả về nằm trong tập action hợp lệ.

Đây là kiểm tra nhanh giao diện Agent, không phải đánh giá sức mạnh chiến thuật.

### 8.5. Cell 4 — Một ván minh họa

```python
demo = play_game(
    RandomAgent(SEED),
    RuleBasedAgent(),
    BOARD_SIZE,
    WIN_LENGTH,
    record=False
)
```

Ván minh họa giúp xác nhận hai agent có thể chơi hết một ván qua tournament engine. `record=False` không lưu toàn bộ chuỗi snapshot bàn cờ trong kết quả.

### 8.6. Cell 5 — Tournament

Notebook định nghĩa bốn cặp:

| Tên cặp | Agent A | Agent B |
|---|---|---|
| Random vs Random | Random A | Random B |
| Rule-Based vs Random | Rule-Based | Random |
| Random vs Rule-Based | Random | Rule-Based |
| Rule-Based vs Rule-Based | Rule-Based A | Rule-Based B |

Mỗi cặp được gọi qua `run_matchup(...)`. Kết quả từng ván được thêm trường `matchup`, ghép thành `results_df` và lưu vào:

```text
results/tables/game_logs.csv
```

Với cấu hình mặc định 20 ván cho mỗi cặp, notebook dự kiến tạo 80 bản ghi tournament nếu tất cả các ván chạy hoàn tất.

### 8.7. Cách tournament cân bằng X/O

Trong `src/tournament.py`, `run_matchup()` luân phiên vị trí:

- Ván chỉ số chẵn: Agent A chơi X, Agent B chơi O.
- Ván chỉ số lẻ: Agent B chơi X, Agent A chơi O.

Khi quy đổi người thắng về A/B, hàm có xét việc hai agent đổi bên. Mục đích là giảm thiên lệch do quyền đi trước.

`play_game()` đo thời gian chọn action riêng cho X và O bằng `time.perf_counter()`, sau đó trả về thời gian trung bình theo vai trò:

- `x_avg_ms`
- `o_avg_ms`

Đây là thời gian đo trên máy thực thi notebook; không nên so sánh trực tiếp giữa các máy khác cấu hình hoặc môi trường Python khác nhau.

### 8.8. Cell 6 — Bảng tổng hợp

Notebook đếm:

- Số ván Agent A thắng.
- Số ván Agent B thắng.
- Số ván hòa.
- Tỷ lệ thắng/hòa theo phần trăm.
- Số nước đi trung bình.
- Thời gian quyết định trung bình của X và O.

Lưu vào:

```text
results/tables/tournament_summary.csv
```

Cách hiểu `A wins` và `B wins`: đây là kết quả theo danh tính agent A/B trong cặp đấu, không phải luôn luôn là X/O, vì vị trí X/O được hoán đổi giữa các ván.

### 8.9. Cell 7 — Phân tích hiệu năng

Notebook nhóm theo `matchup`, tính:

- `games`: số ván.
- `avg_moves`: số nước đi trung bình.
- `avg_x_ms`: thời gian chọn action trung bình của X.
- `avg_o_ms`: thời gian chọn action trung bình của O.

Lưu vào:

```text
results/tables/agent_performance.csv
```

Chú ý: hai cột thời gian X/O là theo vai trò trên bàn cờ, không tự động đại diện cho một agent cố định trong cặp đấu do hai agent đổi bên.

### 8.10. Cell 8 — Biểu đồ

Notebook tạo hai biểu đồ:

1. Biểu đồ tỷ lệ A thắng, B thắng và hòa theo từng matchup.
2. Biểu đồ số nước đi trung bình theo từng matchup.

Lưu tại:

```text
results/charts/win_draw_rates.png
results/charts/average_moves.png
```

### 8.11. Cell 9–10 — Xuất video MP4

Notebook định nghĩa `save_game_mp4(...)`:

- Chạy một ván với `record=True`.
- Lưu snapshot bàn cờ sau từng nước.
- Tạo animation từ các snapshot.
- Sử dụng FFmpeg; nếu không tìm thấy lệnh `ffmpeg` trong PATH, notebook thử lấy FFmpeg từ `imageio_ffmpeg`.
- Ghi video theo FPS được truyền vào (mặc định `fps=2`).

Cell tiếp theo xuất video cho bốn cặp agent:

```text
random_vs_random.mp4
rulebase_vs_random.mp4
random_vs_rulebase.mp4
rulebase_vs_rulebase.mp4
```

Các file được lưu trong `results/videos/`.

Lưu ý: video của cell này là các ván mới được chạy riêng để ghi hình, không phải video được tạo từ các ván tournament đã lưu trong `game_logs.csv`.

### 8.12. Cell kiểm tra kết quả

Liệt kê các file CSV, PNG và MP4 hiện có, kèm kích thước file video. Dùng cell này để xác nhận đầu ra đã được tạo.

### 8.13. Phần Player vs Agent

Notebook có thêm giao diện Human vs AI sử dụng `ipywidgets`.

Theo mã nguồn trong notebook:

- Người chơi là X (`1`), AI là O (`-1`).
- Bàn cờ có 11×11 nút tương ứng 121 ô.
- Có thể chọn Random Agent hoặc Rule-Based Agent.
- Người chơi bấm ô trống để đánh.
- AI chọn action qua `select_action()`.
- Phần giao diện dùng `asyncio.to_thread()` để chạy tính toán agent ngoài event loop, tránh chặn trực tiếp vòng lặp giao diện.
- Có mã trận `round_id` để bỏ qua kết quả AI trả về muộn từ ván cũ.
- Khi đổi agent hoặc nhấn bắt đầu, hàm reset tạo môi trường mới và dựng lại các nút bàn cờ.
- Khi kết thúc, chương trình xuất video MP4 rồi xóa bàn cờ đang hiển thị.
- File MP4 cũ trong `results/videos/` không bị xóa khi reset.

**Lưu ý về Jupyter:** phần giao diện cần chạy trong Jupyter Notebook/JupyterLab có hỗ trợ `ipywidgets`. Nếu đã chạy các phiên bản cell giao diện cũ trước đó, nên Restart Kernel và Clear All Outputs một lần trước khi chạy lại cell giao diện hiện tại để tránh nhiều output/widget cũ cùng tồn tại.

### 8.14. Cell nhận xét và giới hạn

Notebook nêu các nhận xét chính:

- Random Agent phụ thuộc seed và tính ngẫu nhiên.
- Rule-Based Agent dùng heuristic cửa sổ 5 ô.
- Tournament hoán đổi X/O để giảm thiên lệch đi trước.
- Thời gian quyết định phụ thuộc phần cứng và môi trường chạy.
- Rule-Based là heuristic baseline, chưa phải thuật toán tìm kiếm sâu.
- Báo cáo phải dùng số liệu sinh ra từ lần chạy thực tế, không tự suy diễn kết quả.

---

## 9. Tournament engine — `src/tournament.py`

### 9.1. `play_game(...)`

Chức năng: chạy một ván giữa hai agent.

Tham số:

```python
play_game(
    agent_x,
    agent_o,
    board_size=11,
    win_length=5,
    start_player=1,
    record=False
)
```

- `agent_x`: agent chơi X.
- `agent_o`: agent chơi O.
- `board_size`: kích thước bàn.
- `win_length`: số quân liên tiếp cần để thắng.
- `start_player`: người bắt đầu (`1` hoặc `-1`).
- `record`: có lưu snapshot bàn cờ hay không.

Vòng lặp:

1. Kiểm tra `env.is_terminal()`.
2. Lấy người chơi hiện tại.
3. Gọi agent tương ứng để lấy action.
4. Đo thời gian chọn action.
5. Gọi `env.step(action)`.
6. Nếu `record=True`, lưu bản sao bàn cờ.

Kết quả trả về là dictionary gồm winner, số nước, tên agent X/O, thời gian trung bình theo X/O và history nếu bật record. Winner `0` của môi trường được chuẩn hóa thành `None` để biểu thị hòa.

### 9.2. `run_matchup(...)`

Chạy nhiều ván giữa hai factory agent. Mỗi factory nhận seed và trả về một agent mới. Hàm luân phiên A/B ở vị trí X/O và tạo một dòng kết quả cho mỗi ván.

Ví dụ:

```python
from src.tournament import run_matchup
from src.agents.random_agent import RandomAgent
from src.agents.rule_based_agent import RuleBasedAgent

rows = run_matchup(
    lambda seed: RandomAgent(seed=seed, name="Random"),
    lambda seed: RuleBasedAgent(name="Rule-Based"),
    games=20,
    board_size=11,
    win_length=5,
    seed=42
)
```

---

## 10. Cài đặt môi trường

### 10.1. Windows — Conda

Mở Anaconda Prompt hoặc terminal tại thư mục gốc dự án:

```bash
conda create -n caro-ai python=3.11 -y
conda activate caro-ai

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 10.2. Chạy Jupyter

Nếu môi trường chưa có Jupyter Notebook:

```bash
pip install notebook ipywidgets
```

Khởi chạy:

```bash
jupyter notebook
```

Mở:

```text
notebooks/01_rulebase_random.ipynb
```

Nếu dùng JupyterLab, cần môi trường JupyterLab tương thích với `ipywidgets`.

### 10.3. Google Colab

Clone repository từ GitHub:

```python
!git clone <GITHUB_REPOSITORY_URL>
%cd caro_ai_11x11
!pip install -r requirements.txt
```

Sau đó mở notebook hoặc tải notebook lên Colab. Đảm bảo thư mục làm việc hiện tại là thư mục gốc dự án để import `src` chính xác.

### 10.4. Phụ thuộc

`requirements.txt` hiện khai báo:

- NumPy
- pytest
- PyYAML
- Matplotlib
- tqdm
- pandas
- imageio-ffmpeg

Notebook còn sử dụng `ipywidgets` và `imageio`; nếu môi trường chưa có hai gói này, cần cài thêm:

```bash
pip install ipywidgets imageio
```

---

## 11. Cách chạy notebook theo đúng thứ tự

Khuyến nghị:

1. Mở notebook trong thư mục `notebooks/`.
2. Chọn Restart Kernel.
3. Chạy lần lượt từ đầu notebook đến phần tournament.
4. Kiểm tra smoke test trước khi chạy nhiều ván.
5. Chạy tournament.
6. Kiểm tra DataFrame và CSV.
7. Kiểm tra biểu đồ PNG.
8. Chạy các cell xuất MP4.
9. Chạy phần Human vs AI nếu cần demo tương tác.

Nếu chỉ muốn kiểm tra agent nhanh, có thể chạy các phần import/cấu hình, khởi tạo môi trường và smoke test; không nhất thiết chạy tournament hoặc render video.

### 11.1. Chạy thực nghiệm nhanh và chính thức

Notebook mặc định:

```python
GAMES_PER_MATCHUP = 20
```

Để chạy thử nhanh, giữ 20 ván/cặp. Để tăng độ tin cậy thống kê, có thể đổi thành:

```python
GAMES_PER_MATCHUP = 100
```

Sau khi thay đổi, phải chạy lại các cell từ tournament trở đi. Không dùng CSV/biểu đồ cũ để đại diện cho lần chạy mới.

---

## 12. Đầu ra và cách lưu

### 12.1. CSV

| File | Nội dung |
|---|---|
| `results/tables/game_logs.csv` | Kết quả từng ván: matchup, winner, số nước, thời gian, vai trò |
| `results/tables/tournament_summary.csv` | Tổng hợp số trận thắng/thua/hòa và tỷ lệ |
| `results/tables/agent_performance.csv` | Số ván, số nước trung bình và thời gian theo vai trò |

### 12.2. Biểu đồ

| File | Nội dung |
|---|---|
| `results/charts/win_draw_rates.png` | Tỷ lệ thắng của A, B và tỷ lệ hòa |
| `results/charts/average_moves.png` | Số nước đi trung bình theo matchup |

### 12.3. Video

Các file MP4 được lưu trong:

```text
results/videos/
```

Tên video do notebook đặt hoặc do phần Human vs AI tạo. Không xóa thư mục này khi reset một ván nếu cần giữ video các trận trước.

### 12.4. Khả năng tái lập

Để có thể tái lập một thí nghiệm, nên lưu cùng nhau:

- Phiên bản code/commit Git.
- Giá trị `BOARD_SIZE`, `WIN_LENGTH`.
- `GAMES_PER_MATCHUP`.
- Seed.
- Kết quả CSV gốc.
- Biểu đồ sinh ra.
- Phiên bản Python và thư viện.
- Thông tin phần cứng khi báo cáo thời gian chạy.

---

## 13. Kiểm thử

Chạy test từ thư mục gốc:

```bash
pytest -v
```

Notebook smoke test chỉ xác nhận agent trả về action hợp lệ trong trạng thái khởi tạo. Nó không thay thế unit test cho mọi trường hợp.

Các trường hợp nên được kiểm thử riêng:

- Khởi tạo và reset bàn cờ.
- Nước đi hợp lệ và không hợp lệ.
- Đổi lượt.
- Thắng ngang, dọc, chéo xuống, chéo lên.
- Hòa khi bàn đầy.
- Không cho đánh sau khi ván kết thúc.
- `clone()` không chia sẻ ma trận với môi trường gốc.
- `undo()` khôi phục đúng trạng thái.
- Random Agent luôn trả action hợp lệ khi còn nước đi.
- Rule-Based Agent luôn trả action hợp lệ và ưu tiên nước thắng ngay/chặn thắng ngay trong các thế cờ kiểm thử.

---

## 14. Giới hạn và trạng thái triển khai

Các điểm sau cần được hiểu đúng theo gói mã nguồn được cung cấp:

1. Notebook hiện có là `01_rulebase_random.ipynb`; không có nhiều notebook khác trong thư mục `notebooks/` ngoài file này.
2. Hai agent được notebook sử dụng và kiểm thử là Random Agent và Rule-Based Agent.
3. Minimax, MCTS và CNN có file trong `src/agents/`, nhưng notebook hiện tại không dùng chúng trong tournament. Cần đọc mã nguồn và chạy kiểm thử riêng trước khi đưa ra kết luận về hiệu năng.
4. `main.py` và `src/ui_app.py` đang trống trong gói được kiểm tra; README không hướng dẫn chạy chúng như ứng dụng hoàn chỉnh.
5. Thư mục `data/` và `checkpoints/` chưa chứng minh có pipeline huấn luyện hoặc mô hình đã huấn luyện.
6. Kết quả CSV/PNG/MP4 trong `results/` là các artifact đã có trong gói tại thời điểm nhận. Khi chạy notebook lại, các file có thể được tạo mới hoặc ghi đè tùy tên file và cách lưu.
7. Rule-Based Agent là heuristic một bước, không phải Minimax/Alpha-Beta/MCTS và không có bảo đảm tìm được nước đi tối ưu.
8. Số liệu thời gian phụ thuộc máy chạy, Python, tải hệ thống và cấu hình.
9. Muốn đưa kết quả vào báo cáo, cần ghi rõ cấu hình và lấy số liệu từ lần chạy thực tế tương ứng.

---

## 15. Quy trình Git đề xuất

Các lệnh cơ bản:

```bash
git status
git switch dev
git pull
git switch -c docs/readme-notebook-agents
```

Sau khi cập nhật tài liệu:

```bash
git add README.md
git commit -m "docs: document notebook and baseline agents"
git push -u origin docs/readme-notebook-agents
```

Sau đó tạo Pull Request vào nhánh tích hợp theo quy trình nhóm.

Không commit token, mật khẩu, môi trường ảo, cache Python hoặc file dữ liệu lớn nếu nhóm chưa thống nhất.

---

## 16. Tóm tắt nhanh

- Engine: `src/caro_env.py`.
- Agent interface: `BaseAgent.select_action(env)`.
- Random: chọn ngẫu nhiên từ `env.legal_actions()`.
- Rule-Based: đánh giá các cửa sổ 5 ô, thắng ngay, tấn công, phòng thủ và điểm trung tâm.
- Tournament: `src/tournament.py`.
- Notebook thực nghiệm: `notebooks/01_rulebase_random.ipynb`.
- CSV: `results/tables/`.
- Biểu đồ: `results/charts/`.
- Video: `results/videos/`.
- Chạy kiểm thử: `pytest -v`.

Tài liệu này mô tả baseline và notebook hiện có; mọi kết luận hiệu năng phải được xác nhận bằng lần chạy thực nghiệm và dữ liệu đầu ra tương ứng.
