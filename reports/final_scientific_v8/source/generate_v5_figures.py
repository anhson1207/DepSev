#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_v5_figures.py
Generates the 8 priority academic figures for DepSev-ML Scientific Report V5.
- Minimalist academic aesthetics
- Keyword-only concise text (no cluttered sentences)
- Large readable fonts for A4 print
- High resolution (300 DPI)
- Colors: #1A365D (Dark Navy), #2B6CB0 (Steel Blue), #4A5568 (Slate Gray),
          #F7FAFC (Light BG), #C53030 (Red Alert), #DD6B20 (Orange Warning), #2F855A (Green Success)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global matplotlib font settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# Figure 1: Research Progression Pipeline (T5 -> T3)
# ==============================================================================
def make_fig1_research_progression():
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
    ax.axis('off')

    # Background subtle containers
    # Phase 1 Box
    p1 = patches.FancyBboxPatch((0.02, 0.08), 0.44, 0.84, boxstyle="round,pad=0.02",
                                fc='#F8FAFC', ec='#CBD5E1', lw=1.5)
    ax.add_patch(p1)
    ax.text(0.24, 0.88, "GIAI ĐOẠN 1: PHÂN LOẠI 5 MỨC ĐỘ (T5)",
            fontsize=11, fontweight='bold', ha='center', va='center', color='#1E293B')

    # Phase 1 Sub-blocks
    # T5 tiers
    b_t5 = patches.FancyBboxPatch((0.04, 0.58), 0.40, 0.24, boxstyle="round,pad=0.015",
                                 fc='#EFF6FF', ec='#3B82F6', lw=1.2)
    ax.add_patch(b_t5)
    ax.text(0.24, 0.76, "Không gian 5 lớp chuẩn PHQ-9", fontsize=9.5, fontweight='bold', ha='center', color='#1D4ED8')
    ax.text(0.24, 0.65, "C0 (0-4: 74.6%)  |  C1 (5-9: 15.8%)\nC2 (10-14: 6.0%)  |  C3 (15-19: 2.6%)  |  C4 (20-27: 1.1%)",
            fontsize=8.5, ha='center', va='center', color='#334155')

    # T5 Limitations / Bottleneck
    b_lim = patches.FancyBboxPatch((0.04, 0.14), 0.40, 0.38, boxstyle="round,pad=0.015",
                                  fc='#FEF2F2', ec='#EF4444', lw=1.2)
    ax.add_patch(b_lim)
    ax.text(0.24, 0.46, "Rào cản thông tin & Hạn chế", fontsize=9.5, fontweight='bold', ha='center', color='#B91C1C')
    lims = [
        "• Mất cân bằng cực đoan: IR = 68.5:1 (C0 vs C4)",
        "• Chồng lấn sinh lý dày đặc: C0 trùng khít C1 (PCA/IQR)",
        "• Sụp đổ lớp trung gian: Recall C1 chỉ 16.29%",
        "• Chạm trần hiệu năng: Macro-F1 bão hòa quanh 0.27"
    ]
    ax.text(0.06, 0.28, "\n".join(lims), fontsize=8.2, va='center', color='#7F1D1D', linespacing=1.35)

    # Transition Arrow
    ax.annotate('', xy=(0.52, 0.50), xytext=(0.47, 0.50),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=2.5, headwidth=8))
    ax.text(0.495, 0.54, "Tái cấu trúc\nlâm sàng", fontsize=8.5, fontweight='bold', ha='center', color='#334155')

    # Phase 2 Box
    p2 = patches.FancyBboxPatch((0.54, 0.08), 0.44, 0.84, boxstyle="round,pad=0.02",
                                fc='#F8FAFC', ec='#CBD5E1', lw=1.5)
    ax.add_patch(p2)
    ax.text(0.76, 0.88, "GIAI ĐOẠN 2: PHÂN CẤP 2 BƯỚC (T3)",
            fontsize=11, fontweight='bold', ha='center', va='center', color='#1E293B')

    # Stage 1 Box
    b_st1 = patches.FancyBboxPatch((0.56, 0.53), 0.40, 0.29, boxstyle="round,pad=0.015",
                                  fc='#ECFDF5', ec='#10B981', lw=1.2)
    ax.add_patch(b_st1)
    ax.text(0.76, 0.75, "Stage 1: Sàng lọc nhị phân sơ bộ", fontsize=9.5, fontweight='bold', ha='center', color='#047857')
    ax.text(0.76, 0.63, "C0 (0-4: Không triệu chứng)  vs  C12 (5-27: Có triệu chứng)\n• Tập huấn luyện: Toàn bộ đối tượng NHANES\n• Mục tiêu: Nhận diện người có nguy cơ",
            fontsize=8.2, ha='center', va='center', color='#065F46', linespacing=1.3)

    # Arrow Stage 1 -> Stage 2
    ax.annotate('', xy=(0.76, 0.46), xytext=(0.76, 0.52),
                arrowprops=dict(facecolor='#047857', edgecolor='#047857', width=1.5, headwidth=6))

    # Stage 2 Box
    b_st2 = patches.FancyBboxPatch((0.56, 0.14), 0.40, 0.30, boxstyle="round,pad=0.015",
                                  fc='#FFFBEB', ec='#F59E0B', lw=1.2)
    ax.add_patch(b_st2)
    ax.text(0.76, 0.38, "Stage 2: Phân tầng mức độ nghiêm trọng", fontsize=9.5, fontweight='bold', ha='center', color='#B45309')
    ax.text(0.76, 0.25, "C1 (5-9: Nhẹ)  vs  C2+ (10-27: Nặng quy ước)\n• Huấn luyện: CHỈ trên mẫu có nhãn thật PHQ-9 >= 5\n• Cơ chế định tuyến: Hard Routing vs Soft Bayes Risk",
            fontsize=8.2, ha='center', va='center', color='#92400E', linespacing=1.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_research_progression_t5_to_t3.png'), dpi=300)
    plt.close()
    print("Generated: fig_research_progression_t5_to_t3.png")

# ==============================================================================
# Figure 2: Decision Tree Concept Diagram
# ==============================================================================
def make_fig2_decision_tree():
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    ax.axis('off')

    # Root Node
    r_box = patches.FancyBboxPatch((0.35, 0.78), 0.30, 0.16, boxstyle="round,pad=0.02",
                                   fc='#EFF6FF', ec='#1D4ED8', lw=1.5)
    ax.add_patch(r_box)
    ax.text(0.50, 0.88, "NÚT GỐC (Root Node)", fontsize=9.5, fontweight='bold', ha='center', color='#1E40AF')
    ax.text(0.50, 0.82, "Điều kiện: HUQ010 <= 2.5\n(Đánh giá sức khỏe tổng quát)",
            fontsize=8.5, ha='center', color='#334155')

    # Branch left & right from Root
    ax.annotate('', xy=(0.25, 0.65), xytext=(0.42, 0.78),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))
    ax.text(0.31, 0.72, "Đúng (Tốt)", fontsize=8.5, fontweight='bold', color='#047857')

    ax.annotate('', xy=(0.75, 0.65), xytext=(0.58, 0.78),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))
    ax.text(0.67, 0.72, "Sai (Kém)", fontsize=8.5, fontweight='bold', color='#B91C1C')

    # Internal Decision Nodes (Level 1)
    # Left Internal Node
    n_left = patches.FancyBboxPatch((0.10, 0.48), 0.30, 0.16, boxstyle="round,pad=0.02",
                                    fc='#F8FAFC', ec='#64748B', lw=1.2)
    ax.add_patch(n_left)
    ax.text(0.25, 0.58, "Nút quyết định (Internal)", fontsize=9, fontweight='bold', ha='center', color='#334155')
    ax.text(0.25, 0.52, "Điều kiện: SLD010H >= 6.5\n(Thời lượng giấc ngủ)", fontsize=8.2, ha='center', color='#475569')

    # Right Internal Node
    n_right = patches.FancyBboxPatch((0.60, 0.48), 0.30, 0.16, boxstyle="round,pad=0.02",
                                     fc='#F8FAFC', ec='#64748B', lw=1.2)
    ax.add_patch(n_right)
    ax.text(0.75, 0.58, "Nút quyết định (Internal)", fontsize=9, fontweight='bold', ha='center', color='#334155')
    ax.text(0.75, 0.52, "Điều kiện: BMXBMI >= 30.0\n(Chỉ số khối cơ thể)", fontsize=8.2, ha='center', color='#475569')

    # Arrows to leaves
    # From Left Node
    ax.annotate('', xy=(0.14, 0.34), xytext=(0.20, 0.48),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.2, headwidth=5))
    ax.annotate('', xy=(0.36, 0.34), xytext=(0.30, 0.48),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.2, headwidth=5))

    # From Right Node
    ax.annotate('', xy=(0.64, 0.34), xytext=(0.70, 0.48),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.2, headwidth=5))
    ax.annotate('', xy=(0.86, 0.34), xytext=(0.80, 0.48),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.2, headwidth=5))

    # Leaf Nodes (Level 2)
    leaves = [
        (0.04, 0.14, 0.20, "NÚT LÁ 1\nDự đoán: C0\n(Không triệu chứng)\nP(C0) = 0.88", '#ECFDF5', '#10B981', '#065F46'),
        (0.26, 0.14, 0.20, "NÚT LÁ 2\nDự đoán: C1\n(Triệu chứng nhẹ)\nP(C1) = 0.62", '#FFFBEB', '#F59E0B', '#92400E'),
        (0.54, 0.14, 0.20, "NÚT LÁ 3\nDự đoán: C1\n(Triệu chứng nhẹ)\nP(C1) = 0.55", '#FFFBEB', '#F59E0B', '#92400E'),
        (0.76, 0.14, 0.20, "NÚT LÁ 4\nDự đoán: C2+\n(Nặng quy ước)\nP(C2+) = 0.74", '#FEF2F2', '#EF4444', '#7F1D1D')
    ]

    for x, y, w, text, fc, ec, tc in leaves:
        leaf_box = patches.FancyBboxPatch((x, y), w, 0.18, boxstyle="round,pad=0.015",
                                          fc=fc, ec=ec, lw=1.2)
        ax.add_patch(leaf_box)
        ax.text(x + w/2, y + 0.09, text, fontsize=7.8, ha='center', va='center', color=tc, linespacing=1.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_decision_tree_concept.png'), dpi=300)
    plt.close()
    print("Generated: fig2_decision_tree_concept.png")

# ==============================================================================
# Figure 3: Random Forest Architecture
# ==============================================================================
def make_fig3_random_forest_arch():
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    ax.axis('off')

    # Main Dataset
    d_box = patches.FancyBboxPatch((0.02, 0.38), 0.20, 0.30, boxstyle="round,pad=0.02",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(d_box)
    ax.text(0.12, 0.58, "TẬP DỮ LIỆU GỐC", fontsize=9.5, fontweight='bold', ha='center', color='#1E40AF')
    ax.text(0.12, 0.48, "Dữ liệu huấn luyện D\n(N mẫu, d = 34 biến)\nF4 Feature Space",
            fontsize=8.5, ha='center', va='center', color='#334155', linespacing=1.25)

    # Bootstrap Samples
    bootstraps = [
        (0.32, 0.72, "Mẫu Bootstrap 1\n(D1, N mẫu hoàn lại)\nmax_features = 0.5"),
        (0.32, 0.42, "Mẫu Bootstrap 2\n(D2, N mẫu hoàn lại)\nmax_features = 0.5"),
        (0.32, 0.12, "Mẫu Bootstrap B\n(DB, N mẫu hoàn lại)\nmax_features = 0.5")
    ]

    for bx, by, text in bootstraps:
        # Arrow from Main Dataset
        ax.annotate('', xy=(bx, by + 0.08), xytext=(0.22, 0.53),
                    arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.2, headwidth=5))
        b_box = patches.FancyBboxPatch((bx, by), 0.18, 0.18, boxstyle="round,pad=0.015",
                                      fc='#F8FAFC', ec='#64748B', lw=1.2)
        ax.add_patch(b_box)
        ax.text(bx + 0.09, by + 0.09, text, fontsize=7.8, ha='center', va='center', color='#1E293B', linespacing=1.2)

    # Decision Trees (150 trees)
    trees = [
        (0.58, 0.72, "Cây quyết định 1\n(max_depth = 8)\nmin_leaf = 8"),
        (0.58, 0.42, "Cây quyết định 2\n(max_depth = 8)\nmin_leaf = 8"),
        (0.58, 0.12, "Cây quyết định 150\n(max_depth = 8)\nmin_leaf = 8")
    ]

    for (bx, by, _), (tx, ty, ttext) in zip(bootstraps, trees):
        # Arrow from Bootstrap to Tree
        ax.annotate('', xy=(tx, ty + 0.09), xytext=(bx + 0.18, by + 0.09),
                    arrowprops=dict(facecolor='#2563EB', edgecolor='#2563EB', width=1.5, headwidth=5))
        t_box = patches.FancyBboxPatch((tx, ty), 0.18, 0.18, boxstyle="round,pad=0.015",
                                      fc='#ECFDF5', ec='#10B981', lw=1.2)
        ax.add_patch(t_box)
        ax.text(tx + 0.09, ty + 0.09, ttext, fontsize=8, ha='center', va='center', color='#065F46', linespacing=1.2)

    # Dots between Tree 2 and Tree 150
    ax.text(0.41, 0.32, "• • •", fontsize=14, ha='center', va='center', color='#64748B')
    ax.text(0.67, 0.32, "• • •", fontsize=14, ha='center', va='center', color='#64748B')

    # Soft Voting Aggregator
    v_box = patches.FancyBboxPatch((0.82, 0.32), 0.16, 0.42, boxstyle="round,pad=0.02",
                                   fc='#FEF3C7', ec='#F59E0B', lw=1.5)
    ax.add_patch(v_box)
    ax.text(0.90, 0.65, "TỔNG HỢP\nSOFT VOTING", fontsize=9, fontweight='bold', ha='center', color='#92400E')
    ax.text(0.90, 0.47, "Xác suất trung bình:\nbar{P}(c|x) =\n(1/B) sum P_b(c|x)\n\nNhãn cuối cùng:\nargmax bar{P}(c|x)",
            fontsize=7.8, ha='center', va='center', color='#78350F', linespacing=1.2)

    for tx, ty, _ in trees:
        ax.annotate('', xy=(0.82, 0.53), xytext=(tx + 0.18, ty + 0.09),
                    arrowprops=dict(facecolor='#F59E0B', edgecolor='#F59E0B', width=1.2, headwidth=5))

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_random_forest_architecture.png'), dpi=300)
    plt.close()
    print("Generated: fig2_random_forest_architecture.png")

# ==============================================================================
# Figure 4: In-Fold Preprocessing Pipeline & Zero Data Leakage
# ==============================================================================
def make_fig4_infold_cv():
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=300)
    ax.axis('off')

    # NHANES Dataset Box
    d_box = patches.FancyBboxPatch((0.02, 0.36), 0.18, 0.32, boxstyle="round,pad=0.02",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(d_box)
    ax.text(0.11, 0.58, "DỮ LIỆU NHANES", fontsize=9.5, fontweight='bold', ha='center', color='#1E40AF')
    ax.text(0.11, 0.47, "N = 4.045 đối tượng\n34 đặc trưng lâm sàng\nStratified 5-Fold Split\n(Cố định seed = 42)",
            fontsize=8.2, ha='center', va='center', color='#334155', linespacing=1.25)

    # Arrow to CV Fold Split
    ax.annotate('', xy=(0.26, 0.52), xytext=(0.20, 0.52),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    # Inner Processing Box: Single Fold Iteration
    fold_bg = patches.FancyBboxPatch((0.27, 0.10), 0.46, 0.82, boxstyle="round,pad=0.02",
                                     fc='#F8FAFC', ec='#CBD5E1', lw=1.5)
    ax.add_patch(fold_bg)
    ax.text(0.50, 0.86, "LUỒNG XỬ LÝ TRONG MỖI FOLD (ZERO LEAKAGE)",
            fontsize=9.5, fontweight='bold', ha='center', color='#1E293B')

    # Training Fold (80%)
    tr_box = patches.FancyBboxPatch((0.29, 0.50), 0.42, 0.30, boxstyle="round,pad=0.015",
                                    fc='#ECFDF5', ec='#10B981', lw=1.2)
    ax.add_patch(tr_box)
    ax.text(0.50, 0.74, "1. Fold huấn luyện (Training Fold - 80%)", fontsize=8.8, fontweight='bold', ha='center', color='#047857')
    tr_steps = [
        "• Fit Imputer: Tính median / mode CHỈ trên tập train",
        "• Fit Scaler: Tính mean, std (mu, sigma) CHỈ trên train",
        "• Fit Model: Huấn luyện thuật toán với trọng số lớp",
        "-> KHÔNG nhìn thấy bất kỳ thông tin nào của tập test"
    ]
    ax.text(0.31, 0.60, "\n".join(tr_steps), fontsize=7.6, va='center', color='#065F46', linespacing=1.25)

    # Arrow Down inside Fold
    ax.annotate('', xy=(0.50, 0.44), xytext=(0.50, 0.49),
                arrowprops=dict(facecolor='#10B981', edgecolor='#10B981', width=1.5, headwidth=5))
    ax.text(0.51, 0.465, "Chuyển tham số", fontsize=7.5, color='#047857')

    # Test Fold (20%)
    te_box = patches.FancyBboxPatch((0.29, 0.14), 0.42, 0.28, boxstyle="round,pad=0.015",
                                    fc='#FEF2F2', ec='#EF4444', lw=1.2)
    ax.add_patch(te_box)
    ax.text(0.50, 0.36, "2. Fold kiểm tra ngoài mẫu (OOF Test - 20%)", fontsize=8.8, fontweight='bold', ha='center', color='#B91C1C')
    te_steps = [
        "• Transform Imputer: Áp dụng median từ train",
        "• Transform Scaler: Chuẩn hóa bằng mu, sigma từ train",
        "• Predict: Dự đoán nhãn / xác suất khách quan",
        "-> Bảo đảm tính khái quát hóa và trung thực y sinh"
    ]
    ax.text(0.31, 0.23, "\n".join(te_steps), fontsize=7.6, va='center', color='#7F1D1D', linespacing=1.25)

    # Arrow to Evaluation
    ax.annotate('', xy=(0.79, 0.52), xytext=(0.73, 0.52),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    # Output Evaluation Box
    o_box = patches.FancyBboxPatch((0.80, 0.30), 0.18, 0.44, boxstyle="round,pad=0.02",
                                   fc='#FEF3C7', ec='#F59E0B', lw=1.5)
    ax.add_patch(o_box)
    ax.text(0.89, 0.66, "ĐÁNH GIÁ OOF\nTOÀN CỤC", fontsize=9.5, fontweight='bold', ha='center', color='#92400E')
    ax.text(0.89, 0.47, "Gộp dự đoán 5 Folds:\n• Macro-F1\n• Balanced Accuracy\n• Class MAE\n• Ma trận nhầm lẫn\n\n-> Không lạc quan ảo\n-> Tái lập 100%",
            fontsize=8, ha='center', va='center', color='#78350F', linespacing=1.25)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_infold_cv_pipeline.png'), dpi=300)
    plt.close()
    print("Generated: fig3_infold_cv_pipeline.png")

# ==============================================================================
# Figure 5: Redesigned Hard Hierarchical (E1) with Error Propagation Bottleneck
# ==============================================================================
def make_fig5_hard_hierarchical_redesigned():
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    ax.axis('off')

    # Input Sample
    i_box = patches.FancyBboxPatch((0.02, 0.42), 0.16, 0.26, boxstyle="round,pad=0.02",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(i_box)
    ax.text(0.10, 0.58, "ĐỐI TƯỢNG (x)", fontsize=9.5, fontweight='bold', ha='center', color='#1E40AF')
    ax.text(0.10, 0.48, "34 đặc trưng sinh lý\nKhảo sát NHANES\n(N = 4.045)",
            fontsize=8.2, ha='center', va='center', color='#334155', linespacing=1.2)

    # Arrow to Stage 1
    ax.annotate('', xy=(0.24, 0.55), xytext=(0.18, 0.55),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    # Stage 1 Classifier
    st1_box = patches.FancyBboxPatch((0.25, 0.36), 0.24, 0.38, boxstyle="round,pad=0.02",
                                     fc='#ECFDF5', ec='#10B981', lw=1.5)
    ax.add_patch(st1_box)
    ax.text(0.37, 0.67, "GIAI ĐOẠN 1 (M1)", fontsize=10, fontweight='bold', ha='center', color='#047857')
    ax.text(0.37, 0.59, "Sàng lọc nhị phân sơ bộ", fontsize=8.5, fontweight='bold', ha='center', color='#065F46')
    ax.text(0.37, 0.46, "C0 (0-4)  vs  C12 (5-27)\n• Huấn luyện trên toàn bộ N\n• Ngưỡng quyết định: 0.50",
            fontsize=8, ha='center', va='center', color='#065F46', linespacing=1.2)

    # Branch C0 (Stopping)
    ax.annotate('', xy=(0.37, 0.20), xytext=(0.37, 0.36),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=6))
    ax.text(0.38, 0.28, "Dự đoán C0", fontsize=8.5, fontweight='bold', color='#334155')

    stop_box = patches.FancyBboxPatch((0.27, 0.05), 0.20, 0.15, boxstyle="round,pad=0.015",
                                      fc='#F1F5F9', ec='#94A3B8', lw=1.2)
    ax.add_patch(stop_box)
    ax.text(0.37, 0.125, "KẾT THÚC: C0\n(Không triệu chứng)\nDừng xử lý",
            fontsize=8.2, fontweight='bold', ha='center', va='center', color='#475569')

    # Arrow from Stage 1 to Stage 2 (Only if C12 predicted)
    ax.annotate('', xy=(0.57, 0.55), xytext=(0.49, 0.55),
                arrowprops=dict(facecolor='#047857', edgecolor='#047857', width=1.5, headwidth=6))
    ax.text(0.53, 0.58, "Dự đoán C12\n(Có triệu chứng)", fontsize=8, fontweight='bold', ha='center', color='#047857')

    # Stage 2 Classifier
    st2_box = patches.FancyBboxPatch((0.58, 0.36), 0.24, 0.38, boxstyle="round,pad=0.02",
                                     fc='#FFFBEB', ec='#F59E0B', lw=1.5)
    ax.add_patch(st2_box)
    ax.text(0.70, 0.67, "GIAI ĐOẠN 2 (M2)", fontsize=10, fontweight='bold', ha='center', color='#B45309')
    ax.text(0.70, 0.59, "Phân tầng mức độ", fontsize=8.5, fontweight='bold', ha='center', color='#92400E')
    ax.text(0.70, 0.46, "C1 (5-9)  vs  C2+ (10-27)\n• Huấn luyện CHỈ trên mẫu\ncó nhãn thật PHQ-9 >= 5\n(N = 1.029 ca)",
            fontsize=8, ha='center', va='center', color='#92400E', linespacing=1.2)

    # Outcomes from Stage 2
    ax.annotate('', xy=(0.88, 0.65), xytext=(0.82, 0.60),
                arrowprops=dict(facecolor='#F59E0B', edgecolor='#F59E0B', width=1.2, headwidth=5))
    c1_out = patches.FancyBboxPatch((0.88, 0.58), 0.11, 0.14, boxstyle="round,pad=0.015",
                                    fc='#FEF3C7', ec='#F59E0B', lw=1.2)
    ax.add_patch(c1_out)
    ax.text(0.935, 0.65, "NHÃN C1\n(Nhẹ)", fontsize=8.5, fontweight='bold', ha='center', va='center', color='#92400E')

    ax.annotate('', xy=(0.88, 0.45), xytext=(0.82, 0.50),
                arrowprops=dict(facecolor='#EF4444', edgecolor='#EF4444', width=1.2, headwidth=5))
    c2_out = patches.FancyBboxPatch((0.88, 0.38), 0.11, 0.14, boxstyle="round,pad=0.015",
                                    fc='#FEF2F2', ec='#EF4444', lw=1.2)
    ax.add_patch(c2_out)
    ax.text(0.935, 0.45, "NHÃN C2+\n(Nặng quy ước)", fontsize=8.5, fontweight='bold', ha='center', va='center', color='#991B1B')

    # Error Propagation Bottleneck Alert
    err_box = patches.FancyBboxPatch((0.55, 0.05), 0.44, 0.24, boxstyle="round,pad=0.015",
                                     fc='#FEF2F2', ec='#DC2626', lw=1.5, ls='--')
    ax.add_patch(err_box)
    ax.text(0.77, 0.23, "[!] ĐIỂM NGHẼN LAN TRUYỀN SAI SỐ MỘT CHIỀU",
            fontsize=8.8, fontweight='bold', ha='center', color='#B91C1C')
    err_txt = (
        "• Có 120 ca nặng thật sự (C2+) bị M1 đoán nhầm thành C0 ở Giai đoạn 1.\n"
        "• Quyết định cứng khiến 120 ca này bị loại bỏ vĩnh viễn khỏi luồng.\n"
        "• M2 hoàn toàn không có cơ hội tiếp cận -> Bỏ sót ca nặng nguy hiểm!"
    )
    ax.text(0.57, 0.12, err_txt, fontsize=7.8, va='center', color='#7F1D1D', linespacing=1.25)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_4_hard_hierarchical.png'), dpi=300)
    plt.close()
    print("Generated: fig3_4_hard_hierarchical.png")

# ==============================================================================
# Figure 6: Redesigned Soft Hierarchical & Bayes Risk (E4 -> E6)
# ==============================================================================
def make_fig6_soft_hierarchical_bayes():
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=300)
    ax.axis('off')

    # Input Box
    i_box = patches.FancyBboxPatch((0.02, 0.40), 0.14, 0.28, boxstyle="round,pad=0.02",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(i_box)
    ax.text(0.09, 0.58, "ĐỐI TƯỢNG (x)", fontsize=9.2, fontweight='bold', ha='center', color='#1E40AF')
    ax.text(0.09, 0.48, "Đặc trưng sinh lý\n(F4 Space)\nN = 4.045",
            fontsize=8, ha='center', va='center', color='#334155', linespacing=1.2)

    # Arrows to Stage 1 & 2
    ax.annotate('', xy=(0.21, 0.65), xytext=(0.16, 0.57),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))
    ax.annotate('', xy=(0.21, 0.42), xytext=(0.16, 0.51),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    # Stage 1 Soft Model
    st1_box = patches.FancyBboxPatch((0.21, 0.54), 0.22, 0.26, boxstyle="round,pad=0.015",
                                     fc='#ECFDF5', ec='#10B981', lw=1.2)
    ax.add_patch(st1_box)
    ax.text(0.32, 0.74, "Stage 1 (M1 Soft)", fontsize=9, fontweight='bold', ha='center', color='#047857')
    ax.text(0.32, 0.63, "Xuất phân phối xác suất:\n• P_M1(C0 | x)\n• P_M1(C12 | x)",
            fontsize=8, ha='center', va='center', color='#065F46', linespacing=1.2)

    # Stage 2 Soft Model
    st2_box = patches.FancyBboxPatch((0.21, 0.22), 0.22, 0.26, boxstyle="round,pad=0.015",
                                     fc='#FFFBEB', ec='#F59E0B', lw=1.2)
    ax.add_patch(st2_box)
    ax.text(0.32, 0.42, "Stage 2 (M2 Soft)", fontsize=9, fontweight='bold', ha='center', color='#B45309')
    ax.text(0.32, 0.31, "Xuất xác suất điều kiện:\n• P_M2(C1 | C12, x)\n• P_M2(C2+ | C12, x)",
            fontsize=8, ha='center', va='center', color='#92400E', linespacing=1.2)

    # Arrows to Soft Fusion Box
    ax.annotate('', xy=(0.48, 0.58), xytext=(0.43, 0.65),
                arrowprops=dict(facecolor='#047857', edgecolor='#047857', width=1.5, headwidth=6))
    ax.annotate('', xy=(0.48, 0.48), xytext=(0.43, 0.36),
                arrowprops=dict(facecolor='#F59E0B', edgecolor='#F59E0B', width=1.5, headwidth=6))

    # Soft Fusion Box
    fuse_box = patches.FancyBboxPatch((0.48, 0.26), 0.25, 0.54, boxstyle="round,pad=0.02",
                                      fc='#F8FAFC', ec='#64748B', lw=1.5)
    ax.add_patch(fuse_box)
    ax.text(0.605, 0.74, "TÍCH HỢP XÁC SUẤT MỀM", fontsize=9.2, fontweight='bold', ha='center', color='#1E293B')
    f_text = (
        "Xác suất hậu nghiệm toàn thể:\n\n"
        "hat{P}(C0|x) = P_M1(C0|x)\n\n"
        "hat{P}(C1|x) = P_M1(C12|x) *\n"
        "              P_M2(C1|C12,x)\n\n"
        "hat{P}(C2+|x) = P_M1(C12|x) *\n"
        "               P_M2(C2+|C12,x)\n\n"
        "-> Không có quyết định cứng\n"
        "-> Loại bỏ triệt để điểm nghẽn"
    )
    ax.text(0.605, 0.48, f_text, fontsize=7.8, ha='center', va='center', color='#334155', linespacing=1.15)

    # Arrow to Bayes Risk Box
    ax.annotate('', xy=(0.77, 0.53), xytext=(0.73, 0.53),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    # Bayes Risk Minimization Box
    bayes_box = patches.FancyBboxPatch((0.77, 0.16), 0.21, 0.70, boxstyle="round,pad=0.02",
                                       fc='#FEF3C7', ec='#D97706', lw=1.5)
    ax.add_patch(bayes_box)
    ax.text(0.875, 0.80, "QUYẾT ĐỊNH RỦI RO\nBAYES (E6b)", fontsize=9.2, fontweight='bold', ha='center', color='#92400E')
    b_text = (
        "Ma trận chi phí C(i, j):\n"
        "Phạt nặng lỗi bỏ sót:\n"
        "Ma trận chi phí C(j, a):\n"
        "Phạt lỗi bỏ sót nặng:\n"
        "C(2, 0) = 3 (M1) / 5 (M2)\n\n"
        "Hàm rủi ro kỳ vọng:\n"
        "R(a|x) = sum C(j, a) hat{P}_j\n\n"
        "Quy tắc quyết định:\n"
        "y* = argmin_a R(a|x)\n\n"
        "[+] KẾT QUẢ ĐẠT ĐƯỢC:\n"
        "• Bỏ sót C2+ giảm từ\n"
        "  120 ca -> 62 ca (M1)\n"
        "  (Giảm 48.3% bỏ sót!)\n"
        "• BalAcc đạt 50.25%"
    )
    ax.text(0.875, 0.46, b_text, fontsize=7.8, ha='center', va='center', color='#78350F', linespacing=1.18)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_5_soft_cost_sensitive.png'), dpi=300)
    plt.close()
    print("Generated: fig3_5_soft_cost_sensitive.png")

# ==============================================================================
# Figure 7: Algorithm Comparison Bar Chart (T5 and T3)
# ==============================================================================
def make_fig7_algorithm_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Panel A: T5 Algorithms (Nested CV / F4)
    models_t5 = ['LR', 'DT', 'RF (Tuned)', 'XGBoost']
    macro_f1_t5 = [0.2407, 0.2079, 0.2755, 0.2621]
    bal_acc_t5 = [0.3253, 0.2514, 0.2983, 0.2743]
    w_f1_t5 = [0.5599, 0.5898, 0.6437, 0.6196]

    x1 = np.arange(len(models_t5))
    w = 0.26

    r1 = ax1.bar(x1 - w, macro_f1_t5, width=w, label='Macro-F1 (Mục tiêu chính)', color='#2563EB')
    r2 = ax1.bar(x1, bal_acc_t5, width=w, label='Balanced Accuracy', color='#10B981')
    r3 = ax1.bar(x1 + w, w_f1_t5, width=w, label='Weighted F1', color='#94A3B8')

    ax1.set_title('(a) Đối sánh thuật toán trên T5 (5 lớp chuẩn)', fontsize=10.5, fontweight='bold', pad=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(models_t5, fontsize=9.5)
    ax1.set_ylim(0, 0.80)
    ax1.set_ylabel('Điểm số hiệu năng (OOF Mean)', fontsize=9.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8, loc='upper left')

    # Add text labels on top of bars
    for bars in [r1, r2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.2f}",
                     ha='center', va='bottom', fontsize=7.5, rotation=0)

    # Panel B: T3 Paradigms Comparison (E0, E1, E5, E6b, E7a)
    models_t3 = ['E0: Flat RF', 'E1: Hard RF-RF', 'E5: Soft RF-LR', 'E6b: Soft Bayes', 'E7a: Bal OLR']
    macro_f1_t3 = [0.4354, 0.4550, 0.4334, 0.4542, 0.4518]
    bal_acc_t3 = [0.4667, 0.4745, 0.4568, 0.5025, 0.5151]

    x2 = np.arange(len(models_t3))
    w2 = 0.35

    r4 = ax2.bar(x2 - w2/2, macro_f1_t3, width=w2, label='Macro-F1', color='#2563EB')
    r5 = ax2.bar(x2 + w2/2, bal_acc_t3, width=w2, label='Balanced Accuracy', color='#D97706')

    ax2.set_title('(b) Đối sánh các trường phái mô hình hóa trên T3 (3 nhóm)', fontsize=10.5, fontweight='bold', pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['E0 (Flat)', 'E1 (Hard)', 'E5 (Soft)', 'E6b (Bayes)', 'E7a (OLR)'], fontsize=9)
    ax2.set_ylim(0, 0.65)
    ax2.set_ylabel('Điểm số hiệu năng (OOF Mean)', fontsize=9.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8, loc='lower right')

    for bars in [r4, r5]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.2f}",
                     ha='center', va='bottom', fontsize=7.8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_algorithm_comparison_t5_t3.png'), dpi=300)
    plt.close()
    print("Generated: fig4_algorithm_comparison_t5_t3.png")

# ==============================================================================
# Figure 8: Confusion & Error Analysis (Severe FN Bottleneck vs Recovery)
# ==============================================================================
def make_fig8_confusion_error_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Panel A: Severe False Negatives (C2+ misclassified into C0) across models
    models_err = ['E0 (Flat RF)', 'E1 (Hard RF)', 'E4 (Soft RF)', 'E5 (Soft RF-LR)', 'E6b (Bayes M1)', 'E7a (Bal OLR)']
    severe_fn = [108, 120, 227, 211, 62, 68]
    colors_fn = ['#64748B', '#DC2626', '#EA580C', '#3B82F6', '#16A34A', '#8B5CF6']

    b1 = ax1.bar(models_err, severe_fn, color=colors_fn, width=0.55)
    ax1.set_title('(a) Số lượng ca nặng (C2+) bị bỏ sót thành bình thường (C0)', fontsize=10, fontweight='bold', pad=12)
    ax1.set_ylabel('Số ca bệnh bỏ sót (càng thấp càng an toàn)', fontsize=9.5)
    ax1.set_ylim(0, 260)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=25)

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 5, f"{h} ca",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax1.text(0.65, 0.82, "E6b giảm 48.3% ca bỏ sót\nnặng so với E1 (120 -> 62 ca)!\n(Giảm tiếp còn 19 ca với M2)",
             transform=ax1.transAxes, fontsize=8.0, fontweight='bold', color='#15803D',
             bbox=dict(boxstyle="round,pad=0.3", fc="#DCFCE7", ec="#16A34A", lw=1))

    # Panel B: Class-wise Recall comparison: E1 (Hard) vs E6b (Bayes) vs E7a (OLR)
    classes = ['C0 (Bình thường)', 'C1 (Nhẹ)', 'C2+ (Nặng quy ước)']
    rec_e1 = [0.8216, 0.1646, 0.4374]
    rec_e6b = [0.6187, 0.3464, 0.5423]
    rec_e7a = [0.6124, 0.3293, 0.6037]

    x = np.arange(len(classes))
    w = 0.26

    r1 = ax2.bar(x - w, rec_e1, width=w, label='E1: Hard Hierarchical', color='#DC2626')
    r2 = ax2.bar(x, rec_e6b, width=w, label='E6b: Soft Bayes Risk', color='#16A34A')
    r3 = ax2.bar(x + w, rec_e7a, width=w, label='E7a: Balanced OLR', color='#8B5CF6')

    ax2.set_title('(b) So sánh Recall từng lớp: E1 vs E6b vs E7a', fontsize=10, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Tỷ lệ nhận diện (Recall)', fontsize=9.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8, loc='upper right')

    for bars in [r1, r2, r3]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h*100:.1f}%",
                     ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_confusion_error_analysis.png'), dpi=300)
    plt.close()
    print("Generated: fig4_confusion_error_analysis.png")

if __name__ == '__main__':
    print("Starting generation of 8 priority figures for V5...")
    make_fig1_research_progression()
    make_fig2_decision_tree()
    make_fig3_random_forest_arch()
    make_fig4_infold_cv()
    make_fig5_hard_hierarchical_redesigned()
    make_fig6_soft_hierarchical_bayes()
    make_fig7_algorithm_comparison()
    make_fig8_confusion_error_analysis()
    print("All 8 priority figures generated successfully!")

