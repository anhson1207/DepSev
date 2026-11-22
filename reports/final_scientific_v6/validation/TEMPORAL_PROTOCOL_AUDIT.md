# BÁO CÁO KIỂM TOÁN GIAO THỨC KIỂM ĐỊNH DÒNG THỜI GIAN (TEMPORAL VALIDATION AUDIT)
**Dự án:** DepSev-ML (Báo cáo Khoa học V6)  
**Ngày kiểm định:** 30/10/2026  
**Thực hiện:** Ban Biên tập Khoa học V6  

---

## 1. Mục đích kiểm toán
Xác minh tính đúng đắn và sự nhất quán tuyệt đối giữa mã nguồn thực thi, dữ liệu thực nghiệm và văn bản báo cáo về giao thức kiểm định ngoại suy theo dòng thời gian (Branch D: Temporal Generalization).

---

## 2. Phát hiện mâu thuẫn trong Báo cáo V5
Trong Báo cáo V5 (Mục 3.6 và Mục 4.9), văn bản đã mô tả nhầm lẫn giao thức như sau:
> *"Tập huấn luyện gồm toàn bộ chu kỳ NHANES 2013–2014 ($N = 5.054$); Tập kiểm tra độc lập gồm toàn bộ chu kỳ tương lai NHANES 2015–2016 ($N = 4.888$)."*

Mô tả này mâu thuẫn trực tiếp với mã nguồn thực thi trong hai script thực nghiệm chính của dự án:
- `src/15_soft_hierarchical_experiment.py` (dòng 624–633)
- `src/16_ordinal_t3_experiment.py` (dòng 585–594)

---

## 3. Bằng chứng mã nguồn gốc (Ground Truth)

Trích đoạn mã nguồn trong `src/15_soft_hierarchical_experiment.py`:
```python
# 3. Branch D: Temporal Generalization (Train 2011-2014 N=9,693, Test 2015-2016 N=4,888)
print(">>> Running Multi-cycle Branch D (Temporal Generalization N=4,888)...")
train_d = pd.concat([frames["2011_2012"], frames["2013_2014"]], ignore_index=True)
test_d = frames["2015_2016"]
X_tr_d = mp.predictor_frame(train_d, policy)
s_tr_d = train_d.PHQ9_total
y_tr_d = mp.target_values(s_tr_d, "T3")
X_te_d = mp.predictor_frame(test_d, policy)
s_te_d = test_d.PHQ9_total
y_te_d = mp.target_values(s_te_d, "T3")
```

Trích đoạn mã nguồn trong `src/16_ordinal_t3_experiment.py`:
```python
# 3. Branch D: Temporal Generalization (Train 2011-2014 N=9,693, Test 2015-2016 N=4,888)
print(">>> Running Multi-cycle Branch D (Temporal Generalization N=4,888)...")
train_d = pd.concat([frames["2011_2012"], frames["2013_2014"]], ignore_index=True)
test_d = frames["2015_2016"]
```

### Quy mô dữ liệu chính xác:
1. **Tập huấn luyện (Training Set):** Gộp hai chu kỳ khảo sát quá khứ:
   - NHANES 2011–2012 ($N = 4.639$)
   - NHANES 2013–2014 ($N = 5.054$)
   - **Tổng cỡ mẫu huấn luyện:** $N_{\text{train}} = 4.639 + 5.054 = \mathbf{9.693}$ đối tượng.
2. **Tập kiểm tra độc lập (Test Set):** Chu kỳ tương lai:
   - NHANES 2015–2016
   - **Tổng cỡ mẫu kiểm tra:** $N_{\text{test}} = \mathbf{4.888}$ đối tượng.
3. **Không gian đặc trưng sử dụng:**
   - Sử dụng tập 31 biến sinh lý chung ($F_{4\_\text{common31}}$) đã được hài hòa hóa schema qua cả 3 chu kỳ khảo sát CDC NHANES.
4. **Không gian nhãn mục tiêu:**
   - Nhiệm vụ phân tầng 3 nhóm lâm sàng $T_3$ ($C_0$: $0$--$4$, $C_1$: $5$--$9$, $C_2^+$: $10$--$27$).

---

## 4. Kết quả thực nghiệm đã đối soát (Branch D)
Bảng kết quả từ `experiments/ordinal_t3/outputs/multicycle_temporal_results.csv` và `results/tables/multicycle_results.csv`:

| Mô hình | Macro-F1 | Balanced Acc | Accuracy | Recall $C_0$ | Recall $C_1$ | Recall $C_2^+$ | Class MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $E_0$ (Flat RF) | 0.4345 | 0.4908 | 62.68% | 73.66% | 18.79% | 54.77% | 0.5008 |
| $E_1$ (Hard RF–RF) | **0.4393** | 0.4879 | 66.35% | 79.86% | 14.51% | 52.01% | 0.4556 |
| $E_4$ (Soft RF–RF) | 0.4223 | 0.4594 | **71.91%** | 90.60% | 4.76% | 42.46% | **0.3766** |
| $E_5$ (Soft RF–LR) | 0.4265 | 0.4681 | 71.79% | 90.08% | 4.88% | 45.48% | 0.3801 |
| $E_{6b}$ (Soft Cost $M_2$) | 0.3439 | 0.4691 | 34.84% | 26.94% | **60.29%** | 53.52% | 0.7306 |
| **$E_{7a}$ (OLR Balanced)** | 0.4349 | **0.5073** | 56.28% | 61.74% | 31.15% | 59.30% | 0.5503 |
| $E_{7b}$ (CumBin LR) | 0.3632 | 0.4982 | 62.21% | 75.23% | 0.12% | **74.12%** | 0.5833 |
| $E_{7b}$ (CumBin RF) | 0.3861 | 0.4861 | 67.88% | 84.16% | 0.36% | 61.31% | 0.4695 |

---

## 5. Kết luận hiệu chỉnh trong Báo cáo V6
Toàn bộ các vị trí trong Báo cáo V6 đã được sửa đổi và đồng bộ tuyệt đối:
1. **Chương 1 (Mục 1.5, Đóng góp chính):** Ghi rõ tập kiểm định tương lai độc lập NHANES 2015–2016 ($N = 4.888$) sau khi huấn luyện trên chu kỳ 2011–2014 ($N = 9.693$).
2. **Chương 3 (Mục 3.6.2):** Sửa đổi mô tả giao thức Branch D chính xác theo mã nguồn gốc.
3. **Chương 4 (Mục 4.9):** Sửa đổi câu dẫn dắt thực nghiệm phản ánh đúng tập huấn luyện $N = 9.693$ và không gian 31 đặc trưng chung.
4. **Chương 5 (Mục 5.1 & Mục 5.2):** Khẳng định tính khái quát hóa bền vững của mô hình $E_{7a}$ (Balanced OLR) trên tập dữ liệu tương lai.

