# Báo Cáo Tổng Kết Quá Trình Xây Dựng Từ Đầu Đến Cuối — Tác Tử CNN Policy Network & Mixed MCTS (Cấp 5 — Caro 11×11)

Tài liệu này ghi chép đầy đủ toàn bộ quá trình nghiên cứu, thiết kế, lập trình, kiểm thử, tự đánh giá phản biện, khắc phục và hoàn thiện mô-đun **Tác tử Học sâu Có giám sát CNN Policy Network (Cấp 5)** trên nhánh `feat/Neural_Network` của dự án **Caro AI 11×11**.

---

## 1. Bối Cảnh, Cơ Sở Lý Thuyết & Ràng Buộc Dự Án

- **Mục tiêu đề cương (`Mục 1.4 — Cấp 5: Supervised Neural Network Agent`)**:
  - Xây dựng tác tử mạng nơ-ron tích chập (CNN) học có giám sát từ dữ liệu ván đấu tự sinh trên bàn cờ **Caro 11×11** (thắng khi có ít nhất **5 quân liên tiếp** theo hàng ngang, hàng dọc hoặc 2 đường chéo).
  - Tuân thủ tuyệt đối giao diện [`BaseAgent`](../src/agents/base_agent.py) (`select_action(env) -> int` trả về nước đi hợp lệ trong `[0, 120]`), không làm biến đổi trạng thái môi trường [`CaroEnv`](../src/caro_env.py) gốc, và giữ nguyên không chỉnh sửa `src/tournament.py` cũng như `src/ui_app.py`.
- **Tài liệu tham khảo lõi**:
  - **Chương 10 (*Policy Networks in Tic Tac Toe*)** trong sách *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games* (Mark Liu, 2024):
    1. *Mục 10.1.1*: Sinh dữ liệu tự chơi giữa người chơi Chuyên gia (`Expert`) và Bán chuyên (`Non-Expert`, đi ngẫu nhiên 50% để đa dạng hóa trạng thái), chỉ ghi nhãn nước đi của `Expert`.
    2. *Mục 10.1.2*: Biểu diễn trạng thái bàn cờ độc lập người chơi (*Player-Independent Setup*): nhân bàn cờ với `current_player` ($\pm 1$) để 1 mạng duy nhất đánh được cả quân X (`1`) lẫn quân O (`-1`).
    3. *Mục 10.2.1*: Thiết kế 2 biến thể kiến trúc **Fast Policy Network** (`2` lớp ẩn `Dense(64, ReLU)`) và **Strong Policy Network** (`3` lớp ẩn `Dense(64, ReLU)`), huấn luyện bằng `Categorical Cross-Entropy` + `Adam`.
    4. *Mục 10.2.2 & 10.3*: Áp mặt nạ nước đi hợp lệ (*Valid Move Masking*) và chuẩn hóa tổng xác suất $\sum p = 1.0$.
    5. *Mục 10.4.2*: Kết hợp xác suất tiên nghiệm $p_i$ từ mạng CNN vào thuật toán tìm kiếm cây Monte Carlo (**Mixed MCTS** / `mix_mcts`):
       $$\text{mix\_score}_i = v_i + C \sqrt{\frac{\ln N}{n_i}} + \gamma \frac{p_i}{1 + n_i}$$

---

## 2. Toàn Bộ Quá Trình Triển Khai Từ Đầu Đến Cuối (6 Chặng Phát Triển)

### Chặng 1: Khảo sát mã nguồn & Chuẩn hóa hợp đồng tích hợp
- Kiểm tra cấu trúc repository (`src/caro_env.py`, `src/heuristic.py`, `src/agents/base_agent.py`, `configs/default.yaml`, `pytest.ini`, `.github/workflows/ci.yml`) và đối chiếu chuẩn viết code/thực nghiệm với các nhánh của nhóm (`Random`, `RuleBased`, `Minimax`, `MCTS`).
- Xác định hạn chế môi trường: trên một số máy/CI không có TensorFlow hoặc xung đột phiên bản `protobuf` với `tensorflow 2.21`, do đó thiết kế **Dual-Backend (`Keras` + `NumPy` thuần túy)** có cùng kiến trúc toán học và chung định dạng trọng số `.npz`.

### Chặng 2: Xây dựng lõi thuật toán tại [`src/agents/cnn_agent.py`](../src/agents/cnn_agent.py)
Triển khai hoàn chỉnh 4 phân hệ trong `src/agents/cnn_agent.py`:
1. **Mã hóa trạng thái & Tăng cường đối xứng $D_4$**:
   - `encode_board(board, current_player)`: Chuẩn hóa bàn cờ về tensor `(11, 11, 1)` kiểu `float32` (`+1.0` quân mình, `-1.0` quân đối thủ, `0.0` ô trống).
   - `to_one_hot(actions, num_classes=121)`: Chuyển nhãn hành động `0..120` sang vector one-hot `121` chiều.
   - `augment_state_action(state_3d, action)`: Sinh 8 biến thể đối xứng thuộc nhóm dihedral vuông $D_4$ (4 góc quay $0^\circ, 90^\circ, 180^\circ, 270^\circ \times 2$ phép lật ngang) đồng thời cho trạng thái bàn cờ và tọa độ nước đi.
2. **Pipeline tự sinh dữ liệu có giám sát (`Expert` vs `Non-Expert`)**:
   - `expert_select_action(env)`: Tận dụng `get_candidate_actions()` và `evaluate_cell_threat()` trong [`src/heuristic.py`](../src/heuristic.py) theo thứ tự ưu tiên: Thắng ngay 5 quân $\rightarrow$ Chặn thắng 5 quân $\rightarrow$ Tạo chuỗi 4 mở 2 đầu $\rightarrow$ Chặn chuỗi 4 mở 2 đầu / đe dọa kép $\rightarrow$ Điểm Công + Thủ + Vị trí trung tâm.
   - `non_expert_select_action(env, random_prob)`: Đi ngẫu nhiên trong vùng ứng viên theo xác suất `random_prob` và đi nước `Expert` ở phần còn lại.
   - `simulate_expert_vs_non_expert_game(...)`, `generate_expert_dataset(...)`, `generate_expert_train_val_datasets(...)`, `load_expert_train_val_datasets(...)`, và `export_dataset_to_csv(...)`.
3. **Kiến trúc mạng Fast & Strong Policy Network (Dual-Backend)**:
   - `build_keras_policy_network(...)`: Xây dựng mô hình `tensorflow.keras.Sequential` khi môi trường hỗ trợ TensorFlow.
   - `NumpyCNNPolicyNetwork`: Cài đặt đầy đủ `im2col` 2D Convolution (`padding="same"`), `ReLU`, `Flatten`, `2` hoặc `3` lớp `Dense(64, ReLU)`, `Dense(121, Softmax)`, lan truyền ngược đạo hàm `Categorical Cross-Entropy`, điều chuẩn `L2 weight_decay`, và bộ tối ưu `Adam` bằng NumPy thuần túy; đồng bộ trọng số 2 chiều với Keras qua `.npz`.
4. **Lớp tác tử `CNNAgent(BaseAgent)`**:
   - `predict_probs(env)`: Dự đoán xác suất, lọc mặt nạ hợp lệ, hỗ trợ `temperature scaling` và chuẩn hóa tổng xác suất về `1.0`.
   - `_check_tactical_move(env, candidates)`: Phản xạ chiến thuật bắt buộc (bật/tắt qua `use_tactical_shortcut`).
   - `_select_action_mix_mcts(...)` & `_rollout_policy_game(...)`: Thuật toán **Policy-Guided Mixed MCTS** kết hợp UCT với xác suất tiên nghiệm $p_i$ từ mạng CNN.
   - Cung cấp các biến đo đạc (`last_move_time_ms`, `last_search_count`, `last_decision_stage`, `last_action_probs`), hỗ trợ lưu/nạp checkpoint (`save_model`, `load_model`) và giao diện dòng lệnh CLI (`main()`).
   - Đăng ký export `CNNAgent` tại [`src/agents/__init__.py`](../src/agents/__init__.py).

### Chặng 3: Xây dựng bộ kiểm thử tự động [`tests/test_cnn_agent.py`](../tests/test_cnn_agent.py)
- Viết bộ unit test kiểm tra toàn diện: `encode_board`, 8 phép đối xứng $D_4$, `to_one_hot`, sinh dữ liệu tách biệt Train/Val, kiến trúc cả 2 mạng `fast` và `strong`, tổng xác suất hợp lệ bằng `1.0`, ô đã đánh có xác suất `0.0`, bất biến của môi trường `env`, phản xạ thắng/chặn 5 quân, chế độ `Mixed MCTS`, giảm `loss` khi huấn luyện, và lưu/nạp checkpoint `.npz`.
- Toàn bộ bộ test dự án đạt **`66 passed, 1 skipped`** khi chạy `pytest -v` (bao gồm cả `test_dataset_save_load_npz_and_export_csv`).

### Chặng 4: Xây dựng Notebook thực nghiệm độc lập (không dùng `tournament.py`)
- Thiết kế Notebook tự định nghĩa hai hàm `play_game()` và `run_matchup()` ngay trong Notebook (tương đồng cấu trúc `01_rulebase_random.ipynb`), tự động hoán đổi lượt đi trước `X` và đi sau `O` trên bộ 25 thế khai cuộc đối xứng quanh tâm.
- Ghi log ván đấu, tổng hợp `summary_df` (Thắng/Hòa/Thua, Score Rate, CI 95%), `performance_df` (thời gian suy nghĩ `ms`, `search_count`) và vẽ biểu đồ trực quan mà hoàn toàn không tạo hay gọi `src/tournament.py` và `src/ui_app.py`.

### Chặng 5: Tự đánh giá phản biện (Audit) & Khắc phục toàn diện các vấn đề kỹ thuật
Sau khi chạy đợt thực nghiệm đầu tiên, tiến hành rà soát phản biện nghiêm khắc và phát hiện 5 lỗ hổng lớn, từ đó thực hiện nâng cấp triệt để:
1. **Khắc phục rò rỉ dữ liệu Train/Validation (*Data Leakage*)**:
   - *Trước*: Xáo trộn dữ liệu sau khi nhân 8 đối xứng $D_4$ khiến các bản xoay/lật của cùng một thế cờ xuất hiện ở cả Train và Val.
   - *Sau*: Tách biệt **170 ván Train (`seed=42`)** và **30 ván Val độc lập (`seed=100042`)** ở cấp độ ván đấu (*Game-Level Split*) trước khi áp dụng $D_4$ chỉ cho tập Train; mở rộng quy mô từ `60` ván lên **`200` ván**.
2. **Nâng cao năng lực nội tại của mạng Pure CNN (`use_tactical_shortcut=False`)**:
   - *Trước*: Dùng bộ lọc `3×3` (phù hợp Tic Tac Toe) không bao quát hết chuỗi 5 quân và bị overfit nặng, khiến Pure CNN chỉ đạt `12.5%` trước `Non-Expert`.
   - *Sau*: Nâng lên bộ lọc **`Conv2D(64, kernel_size=5×5, padding="same")`** bao quát trọn chuỗi 5 quân theo cả 4 phương + thêm điều chuẩn **`L2 weight_decay = 1.5e-3`**. Kết quả: Pure CNN tăng từ `12.5%` lên **`46.0%–52.0%`** trước `Non-Expert (50%)`, thắng **`95.0%`** trước `Random`, và tự vượt qua **3/4** bài kiểm tra chiến thuật.
3. **Tăng tốc độ Mixed MCTS & kiểm soát giới hạn thời gian `1.0s`**:
   - *Trước*: Mỗi nước Mixed MCTS tốn `~760 ms` và chưa đặt giới hạn thời gian mặc định.
   - *Sau*: Cài đặt hàm `_fast_check_five()`, duy trì tập ứng viên tăng dần (`incremental candidate set`) trong rollout và đặt mặc định `time_limit_sec = 1.0`. Thời gian giảm hơn **20 lần** xuống còn **`31.5 ms/nước`** (ở `24 rollouts`) và **`68.1 ms/nước`** (ở `32 rollouts`).
4. **Tắt cơ chế tự nạp checkpoint ngầm định**:
   - Đổi `auto_load_default=False` mặc định trong `CNNAgent.__init__` để agent khởi tạo mới luôn có trọng số sạch trừ khi truyền rõ `model_path`.
5. **Nâng quy mô thực nghiệm đạt chuẩn đề cương**:
   - Chạy đủ **`100` ván / cặp** (`50` X / `50` O) cho cả **5 cặp đấu chính thức** và **6 cấu hình Ablation Study 1** (gồm cả đối chứng có/không đối xứng $D_4$).
   - Bổ sung đầy đủ **Ablation Study 2B (`num_rollouts`)** bên cạnh **2A ($\gamma$)** (`50` ván / điểm), có tính khoảng tin cậy 95% (`± CI 95%`).

### Chặng 6: Chuẩn hóa tiền tố `04_*`, lưu dữ liệu 2 phiên bản tại 2 nơi & tách log từng cặp đấu
- Đồng bộ toàn bộ tệp của phần CNN về tiền tố **`04_*`** theo thứ tự Notebook thứ 4 (`notebooks/04_cnn_policy_analysis.ipynb`).
- Lưu **2 phiên bản dữ liệu tự sinh** (Phiên bản 1: có đối xứng $D_4$ + Val độc lập; Phiên bản 2: không đối xứng $D_4$) đồng thời tại **2 thư mục**:
  - Định dạng nén tensor `.npz` trong `data/`.
  - Định dạng bảng phẳng `.csv` trong `results/tables/` thông qua `export_dataset_to_csv()`.
- Tách tệp nhật ký 500 ván đấu thành **5 tệp log riêng cho từng cặp đấu** (`100` ván/tệp) trong `results/tables/`.

### Chặng 7: Hoàn thiện khả năng tái lập *Run All* & Nghiệm thu 100%
- Khôi phục đầy đủ nhánh tự sinh metadata, tự huấn luyện 3 mạng (`RETRAIN_MODELS = False` / tự train nếu thiếu file) và luôn vẽ `04_fig_training_curves.png` tại Cell 8 (`cell-4-train`) của [`notebooks/04_cnn_policy_analysis.ipynb`](../notebooks/04_cnn_policy_analysis.ipynb).
- Dọn sạch khối thông báo crash kernel ở Cell 20 (`execution_count = 10`, tổng lỗi `error` outputs toàn Notebook = **`0`**) và đồng bộ 100% số liệu trong Cell 19 khớp tuyệt đối với các bảng CSV mới nhất trên đĩa.
- Bổ sung bài test `test_dataset_save_load_npz_and_export_csv` vào [`tests/test_cnn_agent.py`](../tests/test_cnn_agent.py) (`pytest -v`: **`66 passed, 1 skipped`**).

---

## 3. Danh Mục Toàn Bộ Tệp Đã Xây Dựng & Tạo Ra

### 3.1. Mã nguồn & Kiểm thử
- [`src/agents/cnn_agent.py`](../src/agents/cnn_agent.py) — Lõi thuật toán CNN Policy Network (Cấp 5), sinh/lưu/xuất dữ liệu và Mixed MCTS.
- [`src/agents/__init__.py`](../src/agents/__init__.py) — Export `CNNAgent` trong gói `src.agents`.
- [`tests/test_cnn_agent.py`](../tests/test_cnn_agent.py) — Bộ kiểm thử tự động `pytest` cho `CNNAgent`.
- [`notebooks/04_cnn_policy_analysis.ipynb`](../notebooks/04_cnn_policy_analysis.ipynb) — Notebook thực nghiệm khép kín (không dùng `tournament.py`).

### 3.2. Dữ liệu tự sinh (2 phiên bản tại 2 nơi) & Trọng số huấn luyện (`checkpoints/`)
- **Nơi 1 — Thư mục `data/` (`.npz` nén)**:
  - [`data/04_cnn_expert_train_val_11x11.npz`](../data/04_cnn_expert_train_val_11x11.npz) (`113.1 KB`: `11,384` mẫu `X_train` + `245` mẫu `X_val`).
  - [`data/04_cnn_expert_train_noaug_11x11.npz`](../data/04_cnn_expert_train_noaug_11x11.npz) (`13.4 KB`: `1,423` mẫu `X` không đối xứng).
  - [`data/04_cnn_dataset_metadata.csv`](../data/04_cnn_dataset_metadata.csv) — Bảng mô tả siêu dữ liệu.
- **Nơi 2 — Thư mục `results/tables/` (`.csv` dạng bảng)**:
  - [`results/tables/04_cnn_dataset_train_val_11x11.csv`](../results/tables/04_cnn_dataset_train_val_11x11.csv) (`11,629` dòng).
  - [`results/tables/04_cnn_dataset_train_noaug_11x11.csv`](../results/tables/04_cnn_dataset_train_noaug_11x11.csv) (`1,423` dòng).
  - [`results/tables/04_cnn_dataset_metadata.csv`](../results/tables/04_cnn_dataset_metadata.csv).
- **Trọng số mô hình (`checkpoints/`)**:
  - [`checkpoints/cnn_fast_11x11.npz`](../checkpoints/cnn_fast_11x11.npz) (Fast Policy Network — 2 lớp Dense, có $D_4$).
  - [`checkpoints/cnn_strong_11x11.npz`](../checkpoints/cnn_strong_11x11.npz) (Strong Policy Network — 3 lớp Dense, có $D_4$).
  - [`checkpoints/cnn_strong_noaug_11x11.npz`](../checkpoints/cnn_strong_noaug_11x11.npz) (Strong Policy Network — đối chứng không $D_4$).

### 3.3. Bảng kết quả (`results/tables/04_*.csv`) & Biểu đồ (`results/figures/04_*.png`)
- **Log riêng từng cặp đấu (`100 ván / tệp`, cân bằng 50 X / 50 O)**:
  - [`04_cnn_matchup_1_strong_pure_vs_random_logs.csv`](../results/tables/04_cnn_matchup_1_strong_pure_vs_random_logs.csv)
  - [`04_cnn_matchup_2_fast_hybrid_vs_nonexpert_logs.csv`](../results/tables/04_cnn_matchup_2_fast_hybrid_vs_nonexpert_logs.csv)
  - [`04_cnn_matchup_3_strong_hybrid_vs_nonexpert_logs.csv`](../results/tables/04_cnn_matchup_3_strong_hybrid_vs_nonexpert_logs.csv)
  - [`04_cnn_matchup_4_strong_hybrid_vs_fast_hybrid_logs.csv`](../results/tables/04_cnn_matchup_4_strong_hybrid_vs_fast_hybrid_logs.csv)
  - [`04_cnn_matchup_5_strong_mix_mcts_vs_strong_hybrid_logs.csv`](../results/tables/04_cnn_matchup_5_strong_mix_mcts_vs_strong_hybrid_logs.csv)
- **Bảng tổng hợp & Ablation**:
  - [`04_cnn_game_logs.csv`](../results/tables/04_cnn_game_logs.csv), [`04_cnn_tournament_summary.csv`](../results/tables/04_cnn_tournament_summary.csv), [`04_cnn_agent_performance.csv`](../results/tables/04_cnn_agent_performance.csv)
  - [`04_table_training_history.csv`](../results/tables/04_table_training_history.csv), [`04_table_symmetry_samples.csv`](../results/tables/04_table_symmetry_samples.csv), [`04_table_tactical_checks.csv`](../results/tables/04_table_tactical_checks.csv)
  - [`04_table_ablation_architecture.csv`](../results/tables/04_table_ablation_architecture.csv), [`04_table_gamma_sensitivity.csv`](../results/tables/04_table_gamma_sensitivity.csv), [`04_table_rollouts_sensitivity.csv`](../results/tables/04_table_rollouts_sensitivity.csv), [`04_table_cnn_top_probs.csv`](../results/tables/04_table_cnn_top_probs.csv)
- **7 Biểu đồ trực quan (`results/figures/`)**:
  - [`04_fig_symmetry_augmentation.png`](../results/figures/04_fig_symmetry_augmentation.png)
  - [`04_fig_training_curves.png`](../results/figures/04_fig_training_curves.png)
  - [`04_fig_ablation_architecture.png`](../results/figures/04_fig_ablation_architecture.png)
  - [`04_fig_matchup_win_rates_and_moves.png`](../results/figures/04_fig_matchup_win_rates_and_moves.png)
  - [`04_fig_gamma_sensitivity.png`](../results/figures/04_fig_gamma_sensitivity.png)
  - [`04_fig_rollouts_sensitivity.png`](../results/figures/04_fig_rollouts_sensitivity.png)
  - [`04_fig_cnn_explainability_heatmap.png`](../results/figures/04_fig_cnn_explainability_heatmap.png)

---

## 4. Tóm Tắt Các Kết Quả Thực Nghiệm Chính

1. **Huấn luyện trên tập Validation độc lập (30 ván chưa từng gặp)**:
   - Không đối xứng $D_4$: `30.20%` $\rightarrow$ Có đối xứng $D_4$: **`40.41%`** (`Strong CNN`) và **`42.86%`** (`Fast CNN`) trên 121 lớp hành động (gấp $\sim 50\times$ đoán ngẫu nhiên `0.83%`).
2. **Ablation Study 1 (100 ván / cấu hình trước `Non-Expert 50%`)**:
   - `Strong Pure (No SymAug)`: **`18.0% ± 7.57%`** $\rightarrow$ `Fast Pure (With SymAug)`: **`52.0% ± 9.84%`** & `Strong Pure (With SymAug)`: **`46.0% ± 9.82%`**.
   - `Fast Hybrid (Det)`: **`98.0% ± 2.38%`**, `Strong Hybrid (Stoch)`: **`92.5% ± 5.09%`**, `Strong Hybrid (Det)`: **`98.0% ± 2.76%`**.
3. **5 Cặp đấu chính thức (100 ván / cặp, cân bằng 50 X / 50 O)**:
   - `Strong Pure CNN (No Tactical) vs Random`: **`95.0% ± 4.29%`** (`0.52 ms/nước`).
   - `Fast Hybrid CNN vs Non-Expert (50%)`: **`97.5% ± 2.91%`** (`0.69 ms/nước`).
   - `Strong Hybrid CNN vs Non-Expert (50%)`: **`99.0% ± 1.96%`** (`0.67 ms/nước`).
   - `Strong Hybrid CNN vs Fast Hybrid CNN`: **`44.0% ± 9.37%`** (tương đương trong CI 95%).
   - `Strong CNN + Mixed MCTS vs Strong Hybrid CNN`: **`66.0% ± 9.12%`** (`31.47 ms/nước` — vượt trội có ý nghĩa thống kê).
4. **Ablation Study 2 (`Mixed MCTS`, 50 ván / điểm, `time_limit_sec = 1.0`)**:
   - Độ nhạy `num_rollouts`: `8` (`51.0%`) $\rightarrow$ `16` (`70.0%`) $\rightarrow$ `32` (**`85.0% ± 9.80%`**, `68.1 ms`) $\rightarrow$ `48` (`65.0%`) $\rightarrow$ `64` (`67.0%`, `130.1 ms`).
   - Độ nhạy $\gamma$: Đạt cao nhất ở $\gamma = 2.0$ (**`71.0% ± 12.55%`**) và $\gamma = 10.0$ (`63.0% ± 13.37%`), giảm về `57.0%` khi $\gamma = 20.0$.

---

## 5. Hướng Dẫn Chạy Kiểm Chứng Nhanh

```powershell
# 1. Chạy toàn bộ bộ kiểm thử tự động
pytest -v

# 2. Huấn luyện lại mô hình CNN trực tiếp qua CLI (tùy chọn)
$env:PYTHONPATH=".;src"
python src/agents/cnn_agent.py --model-type strong --games 200 --epochs 12 --batch-size 128 --lr 0.0012
```
