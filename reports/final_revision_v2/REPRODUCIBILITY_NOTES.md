# REPRODUCIBILITY NOTES: DEPSEV-ML FRAMEWORK

**Dự án**: DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Phiên bản Báo cáo**: V2 Final Scientific Revision  
**Ngày cập nhật**: 29/10/2026  

---

## 1. Môi trường Thực thi (Execution Environment)

- **Hệ điều hành**: macOS 15.0+ (hỗ trợ đầy đủ Linux x86_64 và Ubuntu 22.04 LTS).
- **Trình thông dịch Python**: Python 3.13 (`.venv/bin/python`).
- **Các thư viện Python cốt lõi**:
  - `numpy >= 1.26.0`
  - `pandas >= 2.2.0`
  - `scipy >= 1.12.0`
  - `scikit-learn >= 1.4.0`
  - `xgboost >= 2.0.0`
  - `pypdf >= 4.0.0`
  - `matplotlib >= 3.8.0`
  - `seaborn >= 0.13.0`
- **Công cụ biên dịch báo cáo**: `tectonic >= 0.15.0` (Trình biên dịch TeX/LaTeX độc lập hiện đại, tự động tải các gói CTAN cần thiết).

---

## 2. Danh mục các Tập lệnh Thực nghiệm Khả thi (Preserved Scripts)

Các thực nghiệm từ $E_0$ đến $E_7$ được đóng gói hoàn chỉnh trong các tập lệnh Python tại thư mục `src/`:

| Tập lệnh thực thi | Nhóm thực nghiệm phụ trách | Kết quả đầu ra chính |
|---|---|---|
| `src/09_run_hierarchical_classification.py` | $E_0$ (Flat RF), $E_1$ (Hard RF--RF), $E_2$ (Threshold Tuned), $E_{3a}$ (RF--LR), $E_{3b}$ (RF--XGB) | `experiments/hierarchical_t3/outputs/model_comparison.csv`, `fold_metrics.csv`, `stage_metrics.csv` |
| `src/10_run_soft_hierarchical.py` | $E_4$ (Soft RF--RF), $E_5$ (Soft RF--LR), $E_{6a}$ (Soft Cost RF--RF), $E_{6b}$ (Soft Cost RF--LR) | `experiments/soft_hierarchical_t3/outputs/model_comparison.csv`, `cost_tradeoff_summary.csv` |
| `src/11_run_multicycle_hierarchical.py` | Đánh giá tổng quát hóa các mô hình phân cấp trên dữ liệu đa chu kỳ | `experiments/hierarchical_t3/outputs/multicycle_comparison.csv` |
| `src/12_run_ordinal_classification.py` | $E_{7a}$ (Ordinal LogReg Unweighted/Balanced), $E_{7b}$ (Cumulative Binary LR/RF) | `experiments/ordinal_t3/outputs/model_comparison.csv`, `fold_metrics.csv`, ma trận nhầm lẫn |
| `src/13_run_multicycle_ordinal.py` | Đánh giá tổng quát hóa các mô hình thứ bậc trên Branch A, B, D ($N = 4.888$) | `experiments/ordinal_t3/outputs/multicycle_comparison.csv` |
| `src/14_audit_and_signoff.py` | Tập lệnh kiểm tra tự động tính toàn vẹn và ngăn rò rỉ dữ liệu | `results/audit_reports/` |

---

## 3. Trạng thái Tài nguyên Dữ liệu và Kho lưu trữ (Artifact Availability)

### 3.1 Các tài nguyên đã có sẵn đầy đủ trong Repository
- **Toàn bộ kết quả thực nghiệm dạng bảng**: Thư mục `results/tables/` chứa đầy đủ 121 tập tin CSV ghi lại chi tiết out-of-fold predictions, feature inventories, confusion matrices, ablation fold results và SHAP values.
- **Toàn bộ kết quả thực nghiệm $E_0 \to E_7$**: Nằm đầy đủ trong `experiments/hierarchical_t3/`, `experiments/soft_hierarchical_t3/`, `experiments/ordinal_t3/`.
- **Toàn bộ hình ảnh đồ họa độ phân giải cao**: Nằm trong `reports/final_revision_v2/source/figures/`.
- **Đặc tả mô hình chuẩn**: `artifacts/final_model_spec.json`.

### 3.2 Các tài nguyên dữ liệu lớn cần tái tạo khi Clone mới (Fresh Clone)
Do chính sách giới hạn dung lượng của Git và quy định phân phối dữ liệu gốc của CDC/NCHS:
1. `data/processed/cohort.csv`: Không được đưa lên git. Có thể được tái tạo hoàn toàn bằng cách chạy pipeline ingestion từ các file SAS Transport `.XPT` gốc của NHANES hoặc từ bản Kaggle CDC sạch.
2. Các tệp trọng số mô hình `.pkl` và mảng NumPy trung gian `.npz`: Không lưu trữ trong git; toàn bộ tham số, siêu tham số và hạt giống ngẫu nhiên (random seeds) đã được cố định hoàn toàn trong mã nguồn để bảo đảm tính tất định khi tái huấn luyện.

---

## 4. Hướng dẫn Biên dịch Báo cáo PDF từ Mã nguồn

Để biên dịch bản báo cáo hoàn chỉnh từ mã nguồn LaTeX đã sửa đổi:

```bash
# Bước 1: Di chuyển vào thư mục mã nguồn LaTeX
cd reports/final_revision_v2/source

# Bước 2: Chạy trình biên dịch Tectonic
~/.local/bin/tectonic main.tex -o ../

# Bước 3: Sao chép tệp PDF thành tên báo cáo chính thức
cp ../main.pdf ../DepSev_ML_Final_Revised.pdf

# Bước 4: Kiểm tra số trang (chuẩn 70 trang)
python3 -c "import pypdf; print('Pages:', len(pypdf.PdfReader('../DepSev_ML_Final_Revised.pdf').pages))"
```

---

## 5. Hướng dẫn Tái kiểm tra Số liệu Tự động (Automated Number Checking)

Để xác minh rằng tất cả các số liệu trong báo cáo trùng khớp 100% với các bảng CSV gốc:

```bash
# Chạy tập lệnh kiểm tra số liệu
.venv/bin/python reports/final_revision_v2/build_artifacts.py

# Xem bảng kiểm tra kết quả đối chiếu
cat reports/final_revision_v2/validation/numerical_checks.csv
```

Toàn bộ 20/20 chỉ số kiểm tra sẽ xuất hiện trạng thái **PASS**.

