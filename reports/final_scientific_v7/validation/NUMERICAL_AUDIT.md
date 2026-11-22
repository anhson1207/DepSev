# BÁO CÁO KIỂM TOÁN SỐ LIỆU THỰC NGHIỆM (V7)
*(Experimental Numerical Audit and Cross-Verification)*

Tài liệu này xác minh tính chính xác tuyệt đối của mọi số liệu thực nghiệm được trình bày trong báo cáo DepSev-ML V7 đối chiếu với các tệp tin CSV gốc trong kho lưu trữ `results/`.

---

## 1. KIỂM ĐỊNH SỐ CA BỎ SÓT CỦA CẤU HÌNH $E_2$ (HARD RF--RF TUNED)

- **Vấn đề tồn tại ở các phiên bản trước**: Có sự mâu thuẫn giữa 95 ca (Bảng 4.4 cũ), 98 ca (văn bản) và 119 ca (Bảng 4.6 cũ).
- **Tệp dữ liệu gốc**: `results/tables/E2_Hierarchical_RF_OptThresh_confusion_matrix.csv`
- **Dữ liệu thực tế từ CSV**:
  ```csv
  True\Pred,C0,C1,C2+
  C0,2478,316,222
  C1,257,258,123
  C2+,98,82,211
  ```
- **Xác nhận số liệu**:
  - Số ca thực tế là $C_2^+$ bị mô hình dự đoán nhầm thành $C_0$: **98 ca** (chính xác tuyệt đối).
  - Số ca thực tế là $C_0$ bị dự đoán thành $C_2^+$: **222 ca** (khớp chính xác).
  - Tỷ lệ bỏ sót ca nặng tại Stage 1: $98 / 391 = 25.06\%$.
- **Trạng thái trong V7**: Đã thống nhất toàn diện con số **98 ca** trong Bảng 4.4, Bảng 4.6, Bảng 4.8 và toàn bộ các đoạn phân tích.

---

## 2. KIỂM ĐỊNH CỘT $C_0 \to C_2^+$ TRONG BẢNG 4.6 PHẦN B

Bảng đối chiếu giữa các tệp Confusion Matrix gộp (Pooled Out-of-fold) và số liệu Bảng 4.6 trong báo cáo V7:

| Mô hình | Tệp CSV đối soát nguồn | Số liệu $C_0 \to C_2^+$ (V6 cũ) | Số liệu trong Bảng 4.6 (V7) | Số liệu CSV gốc | Trạng thái |
| :--- | :--- | :---: | :---: | :---: | :---: |
| $E_0$ (Flat RF) | `E0_Flat_RF_confusion_matrix.csv` | 147 | **316** | 316 | KHỚP 100% |
| $E_1$ (Hard RF--RF) | `E1_Hierarchical_RF_Default_confusion_matrix.csv` | 111 | **261** | 261 | KHỚP 100% |
| $E_2$ (Hard RF--RF Tuned) | `E2_Hierarchical_RF_OptThresh_confusion_matrix.csv` | 147 | **222** | 222 | KHỚP 100% |
| $E_{3a}$ (Hard XGB--XGB) | `E3a_Hierarchical_XGB_Default_confusion_matrix.csv` | 151 | **300** | 300 | KHỚP 100% |
| $E_{3b}$ (Hard XGB--XGB Tuned) | `E3b_Hierarchical_XGB_OptThresh_confusion_matrix.csv` | 113 | **279** | 279 | KHỚP 100% |
| $E_4$ (Soft RF--RF) | `E4_SoftHierarchical_RF_RF_confusion_matrix.csv` | 139 | **139** | 139 | KHỚP 100% |
| $E_5$ (Soft RF--LR) | `E5_SoftHierarchical_RF_LR_confusion_matrix.csv` | 176 | **176** | 176 | KHỚP 100% |
| $E_{6a}$ (Cost RF--RF $M_1$) | `E6a_CostSensitive_RF_M1_confusion_matrix.csv` | 215 | **215** | 215 | KHỚP 100% |
| $E_{6b}$ (Cost RF--LR $M_1$) | `E6b_CostSensitive_RF_LR_M1_confusion_matrix.csv` | 294 | **294** | 294 | KHỚP 100% |
| $E_{7a}$ (Unweighted OLR) | `E7a_Ordinal_Logistic_Unweighted_confusion_matrix.csv` | 49 | **59** | 59 | KHỚP 100% |
| $E_{7a}$ (Balanced OLR) | `E7a_Ordinal_Logistic_Balanced_confusion_matrix.csv` | 357 | **357** | 357 | KHỚP 100% |
| $E_{7b}$ (CumBin LR) | `E7b_Cumulative_Binary_LR_confusion_matrix.csv` | 727 | **727** | 727 | KHỚP 100% |
| $E_{7b}$ (CumBin RF) | `E7b_Cumulative_Binary_RF_confusion_matrix.csv` | 306 | **313** | 313 | KHỚP 100% |

---

## 3. KIỂM ĐỊNH MA TRẬN NHẦM LẪN 5 LỚP $T_5$ (HÌNH 4.5)

- **Tệp dữ liệu gốc**: `results/tables/granularity_confusion_matrices.csv` (phần tương ứng với `target_granularity == 'T5'`).
- **Dữ liệu phân loại thực tế**:
  - Lớp $C_0$ ($N = 3.016$): Dự đoán đúng 2.295 mẫu (76.09%), nhầm sang $C_1$: 633 mẫu (20.99%), nhầm sang $C_2$: 81 mẫu (2.69%), nhầm sang $C_3$: 7 mẫu (0.23%), nhầm sang $C_4$: 0 mẫu.
  - Lớp $C_1$ ($N = 638$): Dự đoán thành $C_0$: 392 mẫu (61.44%), đúng $C_1$: 101 mẫu (15.83%), nhầm sang $C_2$: 117 mẫu (18.34%), sang $C_3$: 27 mẫu (4.23%), sang $C_4$: 1 mẫu (0.16%).
  - Lớp $C_2$ ($N = 251$): Dự đoán thành $C_0$: 103 mẫu (41.04%), sang $C_1$: 63 mẫu (25.10%), đúng $C_2$: 67 mẫu (26.69%), sang $C_3$: 17 mẫu (6.77%), sang $C_4$: 1 mẫu (0.40%).
  - Lớp $C_3$ ($N = 107$): Dự đoán thành $C_0$: 40 mẫu (37.38%), sang $C_1$: 26 mẫu (24.30%), sang $C_2$: 22 mẫu (20.56%), đúng $C_3$: 19 mẫu (17.76%), sang $C_4$: 0 mẫu.
  - Lớp $C_4$ ($N = 33$): Dự đoán thành $C_0$: 13 mẫu (39.39%), sang $C_1$: 11 mẫu (33.33%), sang $C_2$: 6 mẫu (18.18%), sang $C_3$: 2 mẫu (6.06%), đúng $C_4$: 1 mẫu (3.03%).
- **Xác nhận đồ họa**: Hình ảnh `figures/granularity_t5_confusion_matrix.png` đã trực quan hóa chính xác 100% các giá trị trên kèm theo tỷ lệ phần trăm theo từng hàng.

