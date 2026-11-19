# Báo Cáo Tổng Kết Quá Trình Xây Dựng Từ Đầu Đến Cuối — Tác Tử CNN Policy Network & Mixed MCTS (Cấp 5 — Caro 11×11)

Tài liệu này tổng hợp đầy đủ toàn bộ quá trình nghiên cứu, thiết kế, lập trình, kiểm thử, tự đánh giá phản biện, khắc phục và hoàn thiện mô-đun **Tác tử Học sâu Có giám sát CNN Policy Network (Cấp 5)** trên nhánh `feat/Neural_Network` của dự án **Caro AI 11×11**.

---

## 1. Bối Cảnh, Cơ Sở Lý Thuyết & Ràng Buộc Dự Án

- **Mục tiêu đề cương (`Mục 1.4 — Cấp 5: Supervised Neural Network Agent`)**:
  - Xây dựng tác tử mạng nơ-ron tích chập (CNN) học có giám sát từ dữ liệu ván đấu tự sinh trên bàn cờ **Caro 11×11** (thắng khi có ít nhất **5 quân liên tiếp** theo hàng ngang, hàng dọc hoặc 2 đường chéo).
  - Tuân thủ tuyệt đối giao diện [`BaseAgent`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/base_agent.py) (`select_action(env) -> int` trả về nước đi hợp lệ trong `[0, 120]`), không làm biến đổi trạng thái môi trường [`CaroEnv`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/caro_env.py) gốc, và giữ nguyên không chỉnh sửa `src/tournament.py` cũng như `src/ui_app.py`.
- **Tài liệu tham khảo lõi**:
  - **Chương 10 (*Policy Networks in Tic Tac Toe*)** trong sách *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games* (Mark Liu, 2024):
    1. *Mục 10.1.1*: Sinh dữ liệu tự chơi giữa người chơi Chuyên gia (`Expert`) và Bán chuyên (`Non-Expert`, đi ngẫu nhiên 50% để đa dạng hóa trạng thái), chỉ ghi nhãn nước đi của `Expert`.
    2. *Mục 10.1.2*: Biểu diễn trạng thái bàn cờ độc lập người chơi (*Player-Independent Setup*): nhân bàn cờ với `current_player` ($\pm 1$) để 1 mạng duy nhất đánh được cả quân X (`1`) lẫn quân O (`-1`).
    3. *Mục 10.2.1*: Thiết kế 2 biến thể kiến trúc **Fast Policy Network** (`2` lớp ẩn `Dense(64, ReLU)`) và **Strong Policy Network** (`3` lớp ẩn `Dense(64, ReLU)`), huấn luyện bằng `Categorical Cross-Entropy` + `Adam`.
    4. *Mục 10.2.2 & 10.3*: Áp mặt nạ nước đi hợp lệ (*Valid Move Masking*) và chuẩn hóa tổng xác suất $\sum p = 1.0$.
    5. *Mục 10.4.2*: Kết hợp xác suất tiên nghiệm $p_i$ từ mạng CNN vào thuật toán tìm kiếm cây Monte Carlo (**Mixed MCTS** / `mix_mcts`):
       $$\text{mix\_score}_i = v_i + C \sqrt{\frac{\ln N}{n_i}} + \gamma \frac{p_i}{1 + n_i}$$

---

## 2. Nguồn Gốc Sinh Dữ Liệu & Vị Trí Code Trong Dự Án

- **Dữ liệu tự sinh từ đâu?**
  - Môi trường [`CaroEnv(11, 5)`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/caro_env.py) kết hợp bộ đánh giá đe dọa cục bộ [`get_candidate_actions()` & `evaluate_cell_threat()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/heuristic.py).
  - Trong mỗi ván đấu tự chơi, tác tử Chuyên gia **`expert_select_action()`** (ưu tiên Thắng 5 $\rightarrow$ Chặn 5 $\rightarrow$ Tạo chuỗi 4 mở $\rightarrow$ Chặn chuỗi 4 mở $\rightarrow$ Điểm Công/Thủ/Trung tâm) thi đấu với tác tử Bán chuyên **`non_expert_select_action()`** (đi ngẫu nhiên vùng lân cận với xác suất biến thiên quanh `random_prob = 0.5` để mở rộng không gian trạng thái bàn cờ).
  - Hệ thống **chỉ ghi nhận các cặp `(trạng thái bàn cờ, hành động)` tại lượt đi của `Expert`**, chuẩn hóa góc nhìn người chơi hiện tại (`board * current_player`) và nhân bản qua **8 phép biến đổi đối xứng dihedral $D_4$**.
- **Code sinh và lưu dữ liệu nằm ở đâu?**
  1. Trong [`src/agents/cnn_agent.py`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py):
     - [`encode_board()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py#L46-L62) (dòng 46–62) & [`augment_state_action()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py#L78-L104) (dòng 78–104)
     - [`expert_select_action()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py#L113-L208) (dòng 113–208) & [`non_expert_select_action()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py#L210-L231) (dòng 210–231)
     - [`simulate_expert_vs_non_expert_game()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py#L234-L289) (dòng 234–289)
     - [`generate_expert_dataset()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py#L291-L338) (dòng 291–338)
     - [`generate_expert_train_val_datasets()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py#L341-L411) (dòng 341–411) & [`load_expert_train_val_datasets()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py#L414-L433) (dòng 414–433)
     - [`export_dataset_to_csv()`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py#L436-L492) (dòng 436–492)
  2. Trong [`notebooks/04_cnn_policy_analysis.ipynb`](file:///f:/TTNTNC/Caro/caro_ai_11x11/notebooks/04_cnn_policy_analysis.ipynb) (**Mục 4 — Cell `cell-4-train`**): Gọi hàm sinh dữ liệu, **lưu trực tiếp 2 phiên bản vào thư mục `data/` (`.npz`) và `results/tables/` (`.csv`)** (`data/04_cnn_expert_train_val_11x11.npz`, `data/04_cnn_expert_train_noaug_11x11.npz`, `data/04_cnn_dataset_metadata.csv`, `results/tables/04_cnn_dataset_train_val_11x11.csv`, `results/tables/04_cnn_dataset_train_noaug_11x11.csv`, `results/tables/04_cnn_dataset_metadata.csv`) và kiểm chứng nạp lại từ đĩa.

---

## 3. Toàn Bộ Quá Trình Triển Khai Từ Đầu Đến Cuối (7 Chặng Phát Triển)

### Chặng 1: Khảo sát mã nguồn & Chuẩn hóa hợp đồng tích hợp
- Kiểm tra cấu trúc repository (`src/caro_env.py`, `src/heuristic.py`, `src/agents/base_agent.py`, `configs/default.yaml`, `pytest.ini`, `.github/workflows/ci.yml`) và đối chiếu chuẩn viết code/thực nghiệm với các nhánh của nhóm (`Random`, `RuleBased`, `Minimax`, `MCTS`).
- Thiết kế **Dual-Backend (`Keras` + `NumPy` thuần túy)** có cùng kiến trúc toán học và chung định dạng trọng số `.npz` để chạy ổn định trên mọi môi trường Python/CI lẫn Google Colab GPU.

### Chặng 2: Xây dựng lõi thuật toán tại [`src/agents/cnn_agent.py`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py)
- Cài đặt đầy đủ: `encode_board`, `to_one_hot`, `augment_state_action`, `expert_select_action`, `non_expert_select_action`, `simulate_expert_vs_non_expert_game`, `generate_expert_dataset`, `generate_expert_train_val_datasets`, `load_expert_train_val_datasets`, `export_dataset_to_csv`.
- Cài đặt `NumpyCNNPolicyNetwork` & `build_keras_policy_network` cho 2 kiến trúc **Fast Policy Network** (`2` lớp ẩn `Dense(64, ReLU)`) và **Strong Policy Network** (`3` lớp ẩn `Dense(64, ReLU)`).
- Cài đặt lớp `CNNAgent(BaseAgent)` hỗ trợ cả **Pure Policy** (`use_mix_mcts=False`) lẫn **Policy-Guided Mixed MCTS** (`use_mix_mcts=True`), lưu/nạp trọng số và CLI.
- Đăng ký export `CNNAgent` trong [`src/agents/__init__.py`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/__init__.py).

### Chặng 3: Xây dựng bộ kiểm thử tự động [`tests/test_cnn_agent.py`](file:///f:/TTNTNC/Caro/caro_ai_11x11/tests/test_cnn_agent.py)
- Kiểm thử toàn bộ thành phần mã hóa, đối xứng $D_4$, sinh/lưu/nạp `.npz` và xuất `.csv`, suy luận `fast`/`strong`, mặt nạ xác suất hợp lệ, tính bất biến của `env`, phản xạ chiến thuật, `Mixed MCTS`, huấn luyện giảm `loss` và lưu/nạp checkpoint (`pytest -v`: **`66 passed, 1 skipped`**).

### Chặng 4: Xây dựng Notebook thực nghiệm độc lập (không dùng `tournament.py`)
- Định nghĩa trực tiếp `play_game()` và `run_matchup()` ngay trong Notebook (tương đồng `01_rulebase_random.ipynb`), tự động hoán đổi lượt đi trước `X` và đi sau `O` trên bộ 25 thế khai cuộc đối xứng quanh tâm, giữ nguyên `src/tournament.py` và `src/ui_app.py`.

### Chặng 5: Tự đánh giá phản biện (Audit) & Khắc phục sâu các vấn đề kỹ thuật
1. **Chống rò rỉ dữ liệu Train/Val (*Game-Level Split*)**: Tách biệt **170 ván Train (`seed=42`)** và **30 ván Val độc lập (`seed=100042`)** trước khi áp dụng 8 đối xứng $D_4$ chỉ trên tập Train.
2. **Nâng cấp bộ lọc `Conv2D(64, 5×5)` + L2 `weight_decay = 1.5e-3`**: Bao quát trọn chuỗi thắng 5 quân theo cả 4 phương, tăng Score Rate của `Pure CNN` (tắt luật chiến thuật) trước `Non-Expert (50%)` từ `12.5%` lên **`46.0%–52.0%`**, thắng **`95.0%`** trước `Random`, tự vượt qua **3/4** bài chiến thuật.
3. **Tăng tốc Mixed MCTS & kiểm soát ngân sách `1.0s`**: Dùng `_fast_check_five()` + tập ứng viên tăng dần giảm thời gian từ `~760 ms` xuống **`31.5 ms/nước`** (`24 rollouts`), đặt mặc định `time_limit_sec = 1.0`.
4. **Tắt tự nạp checkpoint ngầm**: Đổi `auto_load_default = False` mặc định trong `CNNAgent.__init__`.
5. **Nâng chuẩn thực nghiệm**: Chạy đủ **`100` ván / cặp** (50 X / 50 O) cho cả 5 cặp đấu chính thức & 6 cấu hình Ablation Study 1; bổ sung đầy đủ Ablation 2A ($\gamma$) và 2B (`num_rollouts`) kèm khoảng tin cậy 95%.

### Chặng 6: Chuẩn hóa tiền tố `04_*`, lưu dữ liệu 2 phiên bản tại 2 nơi & tách log từng cặp đấu
- Đồng bộ toàn bộ tệp CNN về tiền tố **`04_*`** (`notebooks/04_cnn_policy_analysis.ipynb`, 7 hình `04_fig_*.png`, 18 bảng `04_*.csv`, 3 tệp `data/04_*`).
- Lưu 2 phiên bản dữ liệu tự sinh đồng thời ở `data/*.npz` và `results/tables/*.csv`.
- Tách nhật ký 500 ván thành **5 tệp log riêng theo từng cặp đấu** (`04_cnn_matchup_1..5_*_logs.csv`, `100` ván/tệp, cân bằng 50 X / 50 O).

### Chặng 7: Hoàn thiện khả năng tái lập *Run All* & Nghiệm thu 100%
- Khôi phục đầy đủ nhánh tự sinh metadata, tự huấn luyện 3 mạng khi chưa có checkpoint (`RETRAIN_MODELS`) và luôn vẽ `04_fig_training_curves.png` tại Cell 8 (`cell-4-train`).
- Dọn sạch khối thông báo crash kernel ở Cell 20 (`execution_count = 10`, `Error outputs: []`), đồng bộ 100% số liệu trong Cell 19 và thêm unit test `test_dataset_save_load_npz_and_export_csv`.

---

## 4. Danh Mục Tệp Chính & Kết Quả Thực Nghiệm Tóm Tắt

- **Mã nguồn & Notebook**: [`src/agents/cnn_agent.py`](file:///f:/TTNTNC/Caro/caro_ai_11x11/src/agents/cnn_agent.py), [`tests/test_cnn_agent.py`](file:///f:/TTNTNC/Caro/caro_ai_11x11/tests/test_cnn_agent.py), [`notebooks/04_cnn_policy_analysis.ipynb`](file:///f:/TTNTNC/Caro/caro_ai_11x11/notebooks/04_cnn_policy_analysis.ipynb).
- **Dữ liệu tự sinh (2 phiên bản tại 2 nơi)**:
  - [`data/04_cnn_expert_train_val_11x11.npz`](file:///f:/TTNTNC/Caro/caro_ai_11x11/data/04_cnn_expert_train_val_11x11.npz) & [`results/tables/04_cnn_dataset_train_val_11x11.csv`](file:///f:/TTNTNC/Caro/caro_ai_11x11/results/tables/04_cnn_dataset_train_val_11x11.csv) (`11,384` train + `245` val).
  - [`data/04_cnn_expert_train_noaug_11x11.npz`](file:///f:/TTNTNC/Caro/caro_ai_11x11/data/04_cnn_expert_train_noaug_11x11.npz) & [`results/tables/04_cnn_dataset_train_noaug_11x11.csv`](file:///f:/TTNTNC/Caro/caro_ai_11x11/results/tables/04_cnn_dataset_train_noaug_11x11.csv) (`1,423` train không đối xứng).
- **Trọng số (`checkpoints/`)**: [`cnn_fast_11x11.npz`](file:///f:/TTNTNC/Caro/caro_ai_11x11/checkpoints/cnn_fast_11x11.npz), [`cnn_strong_11x11.npz`](file:///f:/TTNTNC/Caro/caro_ai_11x11/checkpoints/cnn_strong_11x11.npz), [`cnn_strong_noaug_11x11.npz`](file:///f:/TTNTNC/Caro/caro_ai_11x11/checkpoints/cnn_strong_noaug_11x11.npz).
- **Số liệu nổi bật (đo trực tiếp trên đĩa)**:
  - **Validation độc lập (`30` ván)**: `NoAug` `30.20%` $\rightarrow$ `Strong (D4)` **`40.41%`**, `Fast (D4)` **`42.86%`**.
  - **Ablation 1 (`100` ván / cấu hình vs `Non-Expert 50%`)**: `Strong Pure (No D4)` **`18.0% ± 7.57%`** $\rightarrow$ `Strong Pure (D4)` **`46.0% ± 9.82%`** & `Fast Pure (D4)` **`52.0% ± 9.84%`** $\rightarrow$ `Hybrid Det` **`98.0% ± 2.38%–2.76%`**.
  - **5 Cặp đấu chính thức (`100` ván / cặp)**: `Strong Pure vs Random` (**`95.0% ± 4.29%`**), `Fast Hybrid vs Non-Expert` (**`97.5% ± 2.91%`**), `Strong Hybrid vs Non-Expert` (**`99.0% ± 1.96%`**), `Strong Hybrid vs Fast Hybrid` (`44.0% ± 9.37%`), `Strong CNN + Mixed MCTS vs Strong Hybrid` (**`66.0% ± 9.12%`**, `31.47 ms/nước`).
  - **Ablation 2 (`50` ván / điểm)**: `num_rollouts` đạt đỉnh **`85.0% ± 9.80%`** tại `32 rollouts` (`68.1 ms/nước`); $\gamma \in [2, 10]$ đạt **`63.0%–71.0%`** và giảm về `57.0%` tại $\gamma = 20$.
