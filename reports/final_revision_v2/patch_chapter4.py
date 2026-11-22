import os, re

file_path = "reports/final_revision_v2/source/chapters/chapter4.tex"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Replace Table 4.1
pattern_t1 = r"\\begin\{table\}\{Kết quả thực nghiệm Feature Ablation.*?\\end\{table\}"
assert re.search(pattern_t1, text, re.DOTALL), "Table 4.1 pattern not found"
text = re.sub(pattern_t1, r"\\input{tables/table_ablation_results}", text, flags=re.DOTALL)

# 2. Update Ablation Analysis
old_abl_analysis = r"""\subsection{Phân tích khoa học}
Từ kết quả Bảng \ref{tab:ablation-results}, một số phát hiện then chốt được rút ra:
\begin{enumerate}
    \item \textbf{Đóng góp vượt bậc của nhóm Cận lâm sàng ($F_1 \to F_2$):} Việc bổ sung các chỉ số xét nghiệm máu (HbA1c, Cholesterol, Albumin, Creatinine) tạo ra bước nhảy vọt lớn nhất về hiệu năng đối với tất cả các mô hình. Macro-F1 của Hồi quy Logistic tăng từ 0.1899 lên 0.2338 (+23.1\%), Random Forest tăng từ 0.2052 lên 0.2350 (+14.5\%), và XGBoost tăng từ 0.2173 lên 0.2447 (+12.6\%). Điều này xác nhận giả thuyết y sinh rằng các rối loạn chuyển hóa toàn thân và phản ứng viêm vi thể có mối liên kết chặt chẽ với các biểu hiện trầm cảm.
    \item \textbf{Đóng góp của nhóm Lối sống và Nhân khẩu học ($F_3, F_4$):} Nhóm đặc trưng lối sống (giấc ngủ, vận động, thuốc lá, rượu bia) và nhân khẩu học (tuổi, giới tính, học vấn, nghèo đói) tiếp tục cải thiện hiệu năng của XGBoost (từ 0.2447 lên 0.2707). Tuy nhiên, đối với Random Forest mặc định, việc tăng từ 19 lên 34 đặc trưng khiến Macro-F1 giảm nhẹ (từ 0.2350 xuống 0.2232), do thuật toán lấy mẫu ngẫu nhiên $\sqrt{d}$ đặc trưng bị phân tán bởi các biến có độ tương quan yếu. Điều này đặt ra yêu cầu cấp thiết phải thực hiện tinh chỉnh siêu tham số (Hyperparameter Tuning).
    \item \textbf{Sự khác biệt giữa Weighted-F1 và Macro-F1:} Weighted-F1 luôn đạt giá trị rất cao (từ 0.55 đến 0.67) do bị chi phối bởi lớp đa số $C_0$ (chiếm 75\%). Ngược lại, Macro-F1 chỉ dao động quanh mức 0.20 -- 0.27, phản ánh chính xác khó khăn thực sự của mô hình trong việc nhận diện các lớp thiểu số $C_2, C_3, C_4$.
\end{enumerate}"""

new_abl_analysis = r"""\subsection{Phân tích khoa học}
Từ kết quả Bảng \ref{tab:ablation-results}, một số phát hiện then chốt được rút ra:
\begin{enumerate}
    \item \textbf{Đóng góp vượt bậc của nhóm Tiền sử bệnh lý lâm sàng ($F_1 \to F_2$):} Theo phân chia nhóm đặc trưng chính thức, nhóm $F_1$ (10 biến) đã bao gồm 5 chỉ số thể chất/sinh tồn và 5 xét nghiệm máu thiết yếu (HbA1c, Cholesterol, HDL, Creatinine, Albumin). Bước chuyển $F_1 \to F_2$ bổ sung 9 biến tiền sử bệnh lý mạn tính và chẩn đoán lâm sàng của bác sĩ (tăng huyết áp, đái tháo đường, 5 bệnh lý tim mạch/đột quỵ, ung thư và tự đánh giá sức khỏe), tạo ra mức tăng hiệu năng rõ nét nhất trên toàn bộ các mô hình: Macro-F1 của Hồi quy Logistic tăng từ 0.1900 lên 0.2338 (+23.1\%), Random Forest tăng từ 0.2052 lên 0.2350 (+14.5\%), và XGBoost tăng từ 0.2173 lên 0.2447 (+12.6\%). Điều này xác nhận giả thuyết y sinh rằng tiền sử bệnh mạn tính cung cấp tín hiệu bổ sung giá trị cho các chỉ số hóa sinh máu.
    \item \textbf{Đóng góp của nhóm Lối sống và Nhân khẩu học ($F_3, F_4$):} Nhóm đặc trưng lối sống (giấc ngủ, vận động thể lực, hút thuốc, sử dụng rượu bia) và nhân khẩu học (tuổi, giới tính, học vấn, tỷ lệ nghèo) tiếp tục nâng hiệu năng của XGBoost lên mức Macro-F1 = \textbf{0.2707 $\pm$ 0.0428} (Balanced Accuracy = 0.2837 $\pm$ 0.0485). Đối với Random Forest chưa tinh chỉnh, việc tăng từ 19 lên 34 đặc trưng khiến Macro-F1 giảm nhẹ (từ 0.2350 xuống 0.2232 $\pm$ 0.0099), do việc lấy mẫu ngẫu nhiên $\sqrt{d}$ đặc trưng tại mỗi nhánh bị phân tán bởi các biến tương quan yếu, đặt ra yêu cầu phải tinh chỉnh siêu tham số có kiểm soát.
    \item \textbf{Sự khác biệt giữa Weighted-F1 và Macro-F1:} Weighted-F1 luôn đạt giá trị rất cao (từ 0.49 đến 0.67) do bị chi phối bởi lớp đa số $C_0$ (chiếm ~75\%). Ngược lại, Macro-F1 dao động quanh mức 0.19 -- 0.27, phản ánh trung thực thách thức trong việc nhận diện các mức độ triệu chứng thiểu số ($C_2, C_3, C_4$).
\end{enumerate}"""
assert old_abl_analysis in text, "Old ablation analysis not found"
text = text.replace(old_abl_analysis, new_abl_analysis)

# 3. Replace Table 4.2
pattern_t2 = r"\\begin\{table\}\{Hiệu năng mô hình theo độ chi tiết nhãn mục tiêu.*?\\end\{table\}"
assert re.search(pattern_t2, text, re.DOTALL), "Table 4.2 pattern not found"
text = re.sub(pattern_t2, r"\\input{tables/table_granularity_results}", text, flags=re.DOTALL)

# 4. Correct E0 Confusion Matrix Description in Section 4.4.2
old_cm_desc = r"Phân tích ma trận nhầm lẫn của $T_3$ (Hình \ref{fig:granularity-t3-cm}) cho thấy: Lớp $C_1$ có tới 436/638 mẫu (68.3\%) bị mô hình nhầm lẫn thành $C_0$, và 98/638 mẫu (15.4\%) bị nhầm thành $C_2^+$. Độ nhạy của lớp $C_1$ chỉ đạt vỏn vẹn \textbf{16.29\%}!"
new_cm_desc = r"Phân tích ma trận nhầm lẫn của $T_3$ (Hình \ref{fig:granularity-t3-cm}) dựa trên kết quả kiểm định Out-of-fold thực tế (\texttt{results/tables/granularity\_oof\_predictions.csv}) cho thấy: Trong tổng số 638 bệnh nhân có triệu chứng nhẹ ($C_1$), có tới \textbf{347 mẫu (54.39\%)} bị mô hình nhầm lẫn thành $C_0$ (bình thường), \textbf{187 mẫu (29.31\%)} bị nhầm thành $C_2^+$ (nặng), và chỉ có \textbf{104 mẫu (16.30\%)} được phân loại chính xác vào lớp $C_1$. Độ nhạy (Recall) của lớp $C_1$ chỉ đạt vỏn vẹn \textbf{16.29\%}!"
assert old_cm_desc in text, "Old confusion matrix description not found"
text = text.replace(old_cm_desc, new_cm_desc)

# 5. Section 4.5: Update Temporal Evaluation text & insert Table 4.7
old_temporal_bullet = r"\item \textbf{Kiểm định thời gian (Branch D):} Trên tập dữ liệu độc lập 2015--2016 ($N = 4.888$), mô hình duy trì Macro-F1 = \textbf{0.2584} và Balanced Accuracy = \textbf{0.3012}, chứng minh mô hình không bị quá khớp theo thời gian nhưng trần hiệu năng phân loại đa mức vẫn tồn tại bất biến."
new_temporal_bullet = r"""\item \textbf{Kiểm định thời gian (Branch D):} Trên tập dữ liệu độc lập NHANES 2015--2016 ($N = 4.888$ đối tượng, gồm đầy đủ tất cả mẫu hợp lệ mà không có bất kỳ mẫu nào bị loại bỏ), mô hình 5 lớp duy trì Macro-F1 = \textbf{0.2584} và Balanced Accuracy = \textbf{0.3012}, chứng minh mô hình không bị quá khớp theo thời gian nhưng sự hạn chế về khả năng phân tách đa mức vẫn tồn tại khách quan."""
assert old_temporal_bullet in text, "Old temporal bullet not found"
text = text.replace(old_temporal_bullet, new_temporal_bullet)

# Also insert Table 4.7 into Section 4.5
old_rq3_conclusion = r"\textbf{Kết luận RQ3:} Việc mở rộng cỡ mẫu đa chu kỳ giúp mô hình ước lượng phân phối của các lớp hiếm ổn định hơn (tăng nhẹ Balanced Accuracy và Recall lớp nặng), nhưng \textbf{không thể phá vỡ trần suy giảm hiệu năng của bài toán phân loại đa mức chi tiết} nếu không gian đặc trưng sinh lý vẫn có độ chồng lấn cao."
new_rq3_conclusion = r"""\textbf{Kết quả kiểm định theo dòng thời gian trên nhiệm vụ 3 lớp $T_3$:}  
Bảng \ref{tab:multicycle-temporal-results} tổng hợp hiệu năng của các mô hình $T_3$ trên tập kiểm định thời gian Branch D ($N = 4.888$). Kết quả cho thấy mô hình Ordinal Logistic Regression cân bằng ($E_{7a}$) tiếp tục duy trì Balanced Accuracy cao nhất (\textbf{0.5073}) và nhận diện được \textbf{59.30\%} số ca bệnh nặng $C_2^+$.

\input{tables/table_multicycle_results}

\textbf{Kết luận RQ3:} Việc mở rộng cỡ mẫu đa chu kỳ giúp mô hình ước lượng phân phối của các lớp hiếm ổn định hơn (tăng nhẹ Balanced Accuracy và Recall lớp nặng), nhưng không thể đảo ngược hoàn toàn sự suy giảm hiệu năng của bài toán phân loại đa mức nếu không gian đặc trưng sinh lý ngoại vi vẫn có độ chồng lấn phân phối cao."""
assert old_rq3_conclusion in text, "Old RQ3 conclusion not found"
text = text.replace(old_rq3_conclusion, new_rq3_conclusion)

# 6. Replace Table 4.3 and update Section 4.6
pattern_t3 = r"\\begin\{table\}\{So sánh hiệu năng giữa Flat Classification và Hard Hierarchical.*?\\end\{table\}"
assert re.search(pattern_t3, text, re.DOTALL), "Table 4.3 pattern not found"
text = re.sub(pattern_t3, r"\\input{tables/table_hard_hierarchical}", text, flags=re.DOTALL)

# Update E2 threshold tuning text and Stage 2 accuracy explanation in Section 4.6
old_e2_text = r"\item \texttt{E2\_Hard\_Tuned}: Hard Hierarchical RF--RF với ngưỡng cắt được tối ưu hóa qua lưới bằng Inner CV ($\tau_1 = 0.40, \tau_2 = 0.50$)."
new_e2_text = r"\item \texttt{E2\_Hard\_Tuned}: Hard Hierarchical RF--RF với ngưỡng cắt được tối ưu hóa tương thích theo từng fold qua Inner CV ($\tau_1 \in [0.45, 0.50], \tau_2 \in [0.50, 0.55]$, trung bình $\tau_1 = 0.47, \tau_2 = 0.53$)."
assert old_e2_text in text, "Old E2 text not found"
text = text.replace(old_e2_text, new_e2_text)

# Add detailed breakdown of Stage 2 accuracy, routed vs oracle, and 5/5 fold wins
old_error_prop = r"""\textbf{Phát hiện về Error Propagation của Hard Routing:}
Mặc dù các mô hình phân cấp $E_1, E_2, E_3a$ đều cải thiện được Macro-F1 so với mô hình phẳng $E_0$ (tăng từ 0.4354 lên 0.4550 -- 0.4586), phân tích luồng dữ liệu chỉ ra một khiếm khuyết lâm sàng nghiêm trọng:
Tại Giai đoạn 1, cổng phân loại nhị phân cứng với ngưỡng $\tau_1 = 0.5$ đã loại bỏ nhầm \textbf{120 bệnh nhân thuộc nhóm Nặng quy ước ($C_2^+$)} (chiếm tới 30.7\% tổng số 391 bệnh nhân nặng). Những bệnh nhân này bị dán nhãn ngay lập tức là $C_0$ (bình thường) và không bao giờ có cơ hội được Giai đoạn 2 xem xét. Đây là minh chứng thực nghiệm điển hình cho hiện tượng lan truyền sai số một chiều trong kiến trúc phân cấp cứng."""

new_error_prop = r"""\textbf{Phân tích chuyên sâu về hiệu năng và Lan truyền sai số của Hard Routing:}
\begin{enumerate}
    \item \textbf{Ưu thế ổn định của $E_1$ so với $E_0$ qua các Folds:} Đối chiếu kết quả kiểm định chéo từng fold (\texttt{experiments/hierarchical\_t3/outputs/fold\_metrics.csv}) cho thấy mô hình Hard Hierarchical $E_1$ giành chiến thắng tuyệt đối trước Flat RF $E_0$ trên \textbf{cả 5/5 folds} (Fold 1: +0.0105; Fold 2: +0.0181; Fold 3: +0.0151; Fold 4: +0.0219; Fold 5: +0.0326), nâng Macro-F1 trung bình từ 0.4354 lên 0.4550 $\pm$ 0.0056.
    \item \textbf{Làm rõ bản chất độ chính xác Giai đoạn 2 (Routed vs Oracle vs End-to-end):} Để tránh hiểu nhầm số liệu thực nghiệm, cần phân biệt rõ ba góc độ đánh giá của bộ phân loại Tầng 2 (\texttt{experiments/hierarchical\_t3/outputs/stage\_metrics.csv}):
    \begin{itemize}
        \item \textit{Routed Evaluation (Độ chính xác thực tế trên luồng điều hướng):} Đạt trung bình \textbf{52.72\%}. Mẫu số của độ đo này là các mẫu được Tầng 1 dự đoán là có triệu chứng (bao gồm cả các ca dương tính thật và các ca báo động nhầm $C_0 \to \text{Symptomatic}$).
        \item \textit{Oracle Evaluation (Độ chính xác lý thuyết nếu Tầng 1 hoàn hảo):} Đạt Balanced Accuracy \textbf{58.69\%}. Đánh giá trên tập mẫu thực sự có triệu chứng ($C_1 \cup C_2^+$, $N = 1.029$).
        \item \textit{End-to-end Evaluation (Độ chính xác toàn hệ thống 3 lớp):} Mô hình $E_1$ đạt \textbf{68.08\%} trên toàn bộ $N = 4.045$ đối tượng.
    \end{itemize}
    \item \textbf{Khiếm khuyết tắc nghẽn lâm sàng của cổng cứng:} Tại Giai đoạn 1, cổng phân loại nhị phân cứng với ngưỡng $\tau_1 = 0.5$ đã loại bỏ nhầm \textbf{120 bệnh nhân thuộc nhóm Nặng quy ước ($C_2^+$)} (chiếm tới 30.7\% tổng số 391 bệnh nhân nặng). Những bệnh nhân này bị dán nhãn ngay lập tức là $C_0$ (bình thường) và không bao giờ có cơ hội được Giai đoạn 2 xem xét. Đây là minh chứng thực nghiệm điển hình cho hiện tượng lan truyền sai số một chiều trong kiến trúc phân cấp cứng.
\end{enumerate}"""
assert old_error_prop in text, "Old error propagation text not found"
text = text.replace(old_error_prop, new_error_prop)

# 7. Replace Table 4.4 and update Section 4.7
pattern_t4 = r"\\begin\{table\}\{Hiệu năng mô hình Soft Hierarchical và Quyết định Nhạy chi phí.*?\\end\{table\}"
assert re.search(pattern_t4, text, re.DOTALL), "Table 4.4 pattern not found"
text = re.sub(pattern_t4, r"\\input{tables/table_soft_hierarchical}", text, flags=re.DOTALL)

# Update Section 4.7 text to discuss E6a
old_soft_analysis = r"""2. \textbf{Tác dụng giải cứu của Ma trận chi phí lâm sàng $M_1$ ($E_6b$):} Khi áp dụng quy tắc rủi ro Bayes với ma trận $M_1$, mô hình $E_6b$ đã kéo số ca bỏ sót nguy hiểm $C_2^+ \to C_0$ từ 211 ca xuống chỉ còn \textbf{62 ca} (giảm 70.6\%), đồng thời nâng Recall $C_2^+$ lên \textbf{54.23\%} và Balanced Accuracy đạt \textbf{0.5025}."""

new_soft_analysis = r"""2. \textbf{Tác dụng giải cứu của Ma trận chi phí lâm sàng $M_1$ ($E_{6a}, E_{6b}$):} Khi áp dụng quy tắc rủi ro Bayes với ma trận $M_1$, cả hai mô hình $E_{6a}$ (RF--RF) và $E_{6b}$ (RF--LR) đều kéo giảm số ca bỏ sót nguy hiểm $C_2^+ \to C_0$ xuống chỉ còn đúng \textbf{62 ca} (giảm hơn 70\% so với mô hình Soft không trọng số). Đặc biệt, mô hình $E_{6a}$ đạt Recall lớp nhẹ $C_1$ lên tới \textbf{40.92\%} (mức cao nhất trong toàn bộ các mô hình thực nghiệm), trong khi $E_{6b}$ đạt Recall lớp nặng $C_2^+$ lên tới \textbf{54.23\%} và Balanced Accuracy đạt \textbf{0.5025}."""
assert old_soft_analysis in text, "Old soft analysis text not found"
text = text.replace(old_soft_analysis, new_soft_analysis)

# 8. Replace Table 4.5 and update Section 4.8
pattern_t5 = r"\\begin\{table\}\{Kết quả thực nghiệm các mô hình Ordinal Classification.*?\\end\{table\}"
assert re.search(pattern_t5, text, re.DOTALL), "Table 4.5 pattern not found"
text = re.sub(pattern_t5, r"\\input{tables/table_ordinal_results}", text, flags=re.DOTALL)

# Correct loss function discussion in Section 4.8.2
old_ord_loss_exp = r"""\textbf{1. Nghịch lý của Class MAE trong E7a không trọng số:}  
Mô hình \texttt{E7a} không trọng số đạt \textbf{Class MAE thấp nhất (0.3377)} và \textbf{Accuracy cao nhất (75.23\%)} trong toàn bộ 10 mô hình của dự án. Tuy nhiên, mô hình hoàn toàn không nhận diện được bất kỳ mẫu nào thuộc lớp $C_1$ (\textbf{Recall $C_1 = 0.00\%$}).  
*Giải thích:* Do hàm mất mát MAE phạt sai lệch theo khoảng cách $|y - \hat{y}|$, việc dự đoán toàn bộ về lớp đa số $C_0$ khiến 74.5\% mẫu có lỗi bằng 0, 15.8\% mẫu $C_1$ chỉ chịu lỗi bằng 1, và 9.7\% mẫu $C_2^+$ chịu lỗi bằng 2. Bộ tối ưu hóa tự nhiên triệt tiêu lớp trung gian $C_1$ để tối thiểu hóa MAE kỳ vọng. Điều này chứng minh rằng: \textbf{Chỉ dựa vào Class MAE thấp để kết luận mô hình thứ bậc tốt là một sai lầm nghiêm trọng trên dữ liệu mất cân bằng.}"""

new_ord_loss_exp = r"""\textbf{1. Nghịch lý của Class MAE trong E7a không trọng số:}  
Mô hình \texttt{E7a} không trọng số đạt \textbf{Class MAE thấp nhất (0.3377)} và \textbf{Accuracy cao nhất (75.23\%)} trong toàn bộ các mô hình của dự án. Tuy nhiên, mô hình hoàn toàn không nhận diện được bất kỳ mẫu nào thuộc lớp $C_1$ (\textbf{Recall $C_1 = 0.00\%$}).  
*Làm rõ cơ chế tối ưu hóa:* Về mặt toán học, mô hình $E_{7a}$ tối ưu hóa hàm Negative Log-Likelihood với điều chuẩn $L_2$ trên các trọng số $\mathbf{w}$ (Negative Log-Likelihood của mô hình logit tích lũy), chứ không phải tối thiểu hóa trực tiếp Class MAE trong quá trình huấn luyện. Class MAE là chỉ số đánh giá hình học hậu kỳ. Dưới hiện tượng mất cân bằng lớp cực đoan (lớp $C_0$ chiếm ~75\%), nghiệm cực đại hợp lý (Maximum Likelihood) tự nhiên đẩy các ngưỡng cắt $\theta_0, \theta_1$ sao cho xác suất tích lũy dồn phần lớn về lớp đa số $C_0$. Khi đó, 74.5\% mẫu có khoảng cách lỗi bằng 0, 15.8\% mẫu $C_1$ chỉ chịu lỗi bằng 1, và 9.7\% mẫu $C_2^+$ chịu lỗi bằng 2, dẫn tới giá trị kỳ vọng của Class MAE hậu kỳ đạt mức rất thấp. Điều này chứng minh rằng: \textbf{Chỉ dựa vào Class MAE thấp để kết luận mô hình thứ bậc tốt là một sai lầm nghiêm trọng trên dữ liệu y tế mất cân bằng nếu không kiểm tra độ nhạy từng lớp.}"""
assert old_ord_loss_exp in text, "Old ordinal loss explanation not found"
text = text.replace(old_ord_loss_exp, new_ord_loss_exp)

# 9. Replace Table 4.6
pattern_t6 = r"\\begin\{table\}\{Bảng so sánh tổng hợp toàn diện các mô hình.*?\\end\{table\}"
assert re.search(pattern_t6, text, re.DOTALL), "Table 4.6 pattern not found"
text = re.sub(pattern_t6, r"\\input{tables/table_master_comparison}", text, flags=re.DOTALL)

# 10. Update Section 4.10 & 4.11 tone
old_ceiling_claim = r"""Hiện tượng diện tích chồng lấn Bayes $\Omega_{\text{overlap}}$ giữa người bình thường ($C_0$) và người trầm cảm nhẹ ($C_1$) rất lớn (Hình \ref{fig:overlap-pca}) tạo nên một \textbf{trần hiệu năng tự nhiên (Performance Ceiling)} trong khoảng Macro-F1 từ $0.45$ đến $0.46$. Không có bất kỳ thuật toán nào có thể vượt qua trần hiệu năng này nếu chỉ thuần túy sử dụng các biến số sinh lý hiện có mà không bổ sung các dữ liệu chuyên sâu hơn (như gen, nồng độ Cortisol huyết thanh, hoặc dữ liệu hành vi di động liên tục)."""

new_ceiling_claim = r"""Hiện tượng diện tích chồng lấn Bayes $\Omega_{\text{overlap}}$ giữa người bình thường ($C_0$) và người trầm cảm nhẹ ($C_1$) rất lớn (Hình \ref{fig:overlap-pca}) tạo nên một rào cản hiệu năng đối với các bộ phân loại hiện tại trong khoảng Macro-F1 từ $0.45$ đến $0.46$. Trong giới hạn của các biến số sinh lý ngoại vi và xét nghiệm máu thường quy hiện có trong NHANES, việc nâng cao hơn nữa hiệu năng phân tách đòi hỏi phải bổ sung các chiều dữ liệu y sinh chuyên sâu hơn (như các chỉ dấu phân tử sinh học, hormone căng thẳng Cortisol hoặc dữ liệu theo dõi hành vi liên tục từ thiết bị đeo)."""
assert old_ceiling_claim in text, "Old ceiling claim not found"
text = text.replace(old_ceiling_claim, new_ceiling_claim)

old_rq5 = r"""\item \textbf{Trả lời RQ5:} Mô hình Soft Hierarchical nhạy chi phí ($E_6b$) và Ordinal Logistic Regression cân bằng ($E_7a$) đã giải quyết triệt để sự tắc nghẽn của cổng cứng, giảm thiểu số ca bỏ sót nặng xuống còn 62--68 ca, thiết lập kỷ lục Balanced Accuracy 0.5151 và duy trì độ bền vững trên tập kiểm tra theo dòng thời gian (Branch D, Balanced Accuracy = 0.5073)."""
new_rq5 = r"""\item \textbf{Trả lời RQ5:} Mô hình Soft Hierarchical nhạy chi phí ($E_{6a}, E_{6b}$) và Ordinal Logistic Regression cân bằng ($E_{7a}$) đã khắc phục hiệu quả sự tắc nghẽn của cổng cứng, giảm số ca bỏ sót nặng $C_2^+ \to C_0$ từ 120 ca xuống còn 62--68 ca, nâng Balanced Accuracy lên mức cao nhất 0.5151 và duy trì độ ổn định trên tập kiểm tra theo dòng thời gian (Branch D, Balanced Accuracy = 0.5073)."""
assert old_rq5 in text, "Old RQ5 not found"
text = text.replace(old_rq5, new_rq5)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Chapter 4 updated successfully!")

