#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-grade, 300 DPI scientific figures for the DepSev-ML thesis.
All text in diagrams is formatted clearly in Vietnamese and professional academic styling.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Configure matplotlib for crisp publication quality
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Figure 1: Problem Formulation Diagram
# -------------------------------------------------------------
def make_fig1_problem_formulation():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')

    # Draw 4 main blocks: Input, Data Engine, ML Modeling, Risk Stratification
    # 1. Input Features
    box1 = patches.FancyBboxPatch((0.02, 0.15), 0.22, 0.70, boxstyle="round,pad=0.03",
                                  fc='#EBF3FB', ec='#2B6CB0', lw=1.5)
    ax.add_patch(box1)
    ax.text(0.13, 0.80, "DỮ LIỆU SINH LÝ\nVÀ LÂM SÀNG", weight='bold', fontsize=11,
            ha='center', va='center', color='#1A365D')
    inputs = [
        "• Nhân trắc & Dấu hiệu sinh tồn:\n  BMI, Vòng eo, Huyết áp, Nhịp tim",
        "• Hóa sinh máu:\n  HbA1c, Cholesterol, HDL,\n  Creatinine, Albumin",
        "• Tiền sử bệnh & Lối sống:\n  Bệnh mạn tính, Vận động,\n  Giấc ngủ, Rượu bia, Thuốc lá",
        "• Nhân khẩu học:\n  Tuổi, Giới tính, Giáo dục, PIR"
    ]
    ax.text(0.04, 0.44, "\n\n".join(inputs), fontsize=8.5, va='center', color='#2D3748', linespacing=1.2)

    # Arrow 1 -> 2
    ax.annotate('', xy=(0.28, 0.50), xytext=(0.245, 0.50),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=7))

    # 2. Data Preprocessing & Leakage Control
    box2 = patches.FancyBboxPatch((0.28, 0.20), 0.20, 0.60, boxstyle="round,pad=0.03",
                                  fc='#FEFCBF', ec='#B7791F', lw=1.5)
    ax.add_patch(box2)
    ax.text(0.38, 0.74, "KHUNG XỬ LÝ\nCHUẨN TẮC", weight='bold', fontsize=10.5,
            ha='center', va='center', color='#744210')
    engine_text = [
        "• In-fold Imputation\n  (Median / Mode)",
        "• Robust / Standard\n  Scaling (Không rò rỉ)",
        "• One-Hot Encoding",
        "• Stratified 5-Fold\n  Partitioning (SHA-256)",
        "• Multi-cycle\n  Harmonization (31 biến)"
    ]
    ax.text(0.295, 0.44, "\n\n".join(engine_text), fontsize=8.5, va='center', color='#744210', linespacing=1.2)

    # Arrow 2 -> 3
    ax.annotate('', xy=(0.52, 0.50), xytext=(0.485, 0.50),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=7))

    # 3. Multi-Paradigm Machine Learning
    box3 = patches.FancyBboxPatch((0.52, 0.15), 0.22, 0.70, boxstyle="round,pad=0.03",
                                  fc='#E6FFFA', ec='#2C7A7B', lw=1.5)
    ax.add_patch(box3)
    ax.text(0.63, 0.80, "ĐA TRƯỜNG PHÁI\nMÔ HÌNH HÓA", weight='bold', fontsize=11,
            ha='center', va='center', color='#234E52')
    models_text = [
        "1. Phân loại phẳng (Flat):\n   RF, XGBoost, LR",
        "2. Phân loại phân cấp\n   (Hierarchical):\n   • Hard Routing (E1-E3)\n   • Soft Probability (E4)\n   • Bayes Cost-Sensitive (E5-E6)",
        "3. Phân loại thứ bậc\n   (Ordinal):\n   • Ordinal Logistic (E7a)\n   • Cumulative Binary (E7b)"
    ]
    ax.text(0.535, 0.44, "\n\n".join(models_text), fontsize=8.5, va='center', color='#234E52', linespacing=1.2)

    # Arrow 3 -> 4
    ax.annotate('', xy=(0.78, 0.50), xytext=(0.745, 0.50),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=7))

    # 4. Clinical Severity Risk Tiers
    box4 = patches.FancyBboxPatch((0.78, 0.15), 0.20, 0.70, boxstyle="round,pad=0.03",
                                  fc='#FED7D7', ec='#C53030', lw=1.5)
    ax.add_patch(box4)
    ax.text(0.88, 0.80, "PHÂN TẦNG NGUY CƠ\nTRIỆU CHỨNG PHQ-9", weight='bold', fontsize=10.5,
            ha='center', va='center', color='#742A2A')
    outputs = [
        "Không gian T5 (5 bậc):\n• C0: Tối thiểu (0-4)\n• C1: Nhẹ (5-9)\n• C2: Vừa (10-14)\n• C3: Nặng vừa (15-19)\n• C4: Rất nặng (20-27)",
        "Không gian T3 (3 nhóm):\n• C0: Bình thường (0-4)\n• C1: Nhẹ (5-9)\n• C2+: Nặng quy ước (10-27)"
    ]
    ax.text(0.795, 0.44, "\n\n".join(outputs), fontsize=8.5, va='center', color='#742A2A', linespacing=1.2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig1_problem_formulation.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 2. Figure 2: PHQ-9 Scale & Severity Tiers
# -------------------------------------------------------------
def make_fig2_phq9_tiers():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    
    # 5 Tiers
    tiers = [
        {"name": "C0: Tối thiểu\n(Minimal)", "range": (0, 4.9), "color": "#48BB78", "action": "Không cần can thiệp y tế"},
        {"name": "C1: Nhẹ\n(Mild)", "range": (5, 9.9), "color": "#ECC94B", "action": "Tư vấn lối sống & theo dõi"},
        {"name": "C2: Vừa\n(Moderate)", "range": (10, 14.9), "color": "#ED8936", "action": "Đánh giá tâm thần, cân nhắc trị liệu"},
        {"name": "C3: Nặng vừa\n(Moderately Severe)", "range": (15, 19.9), "color": "#DD6B20", "action": "Can thiệp dược lý và tâm lý tích cực"},
        {"name": "C4: Rất nặng\n(Severe)", "range": (20, 27.0), "color": "#E53E3E", "action": "Yêu cầu can thiệp y tế khẩn cấp"}
    ]
    
    for t in tiers:
        x0, x1 = t["range"]
        rect = patches.Rectangle((x0, 1.2), x1 - x0, 1.2, facecolor=t["color"], edgecolor='white', lw=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text((x0 + x1)/2, 1.8, t["name"], ha='center', va='center', weight='bold', fontsize=9, color='white' if t["color"] != '#ECC94B' else '#2D3748')
        ax.text((x0 + x1)/2, 0.7, t["action"], ha='center', va='center', fontsize=7.5, color='#4A5568', style='italic')

    # Add Scale Axis
    ax.plot([0, 27], [1.2, 1.2], color='#2D3748', lw=2)
    for tick in [0, 5, 10, 15, 20, 27]:
        ax.plot([tick, tick], [1.1, 1.3], color='#2D3748', lw=1.5)
        ax.text(tick, 0.95, str(tick), ha='center', va='center', weight='bold', fontsize=9.5)

    # Brackets for T3 mapping
    # C0
    ax.plot([0, 0, 4.9, 4.9], [2.7, 2.9, 2.9, 2.7], color='#2B6CB0', lw=1.5)
    ax.text(2.45, 3.1, "T3: Nhóm C0 (Tối thiểu)", ha='center', va='bottom', fontsize=8.5, weight='bold', color='#2B6CB0')

    # C1
    ax.plot([5, 5, 9.9, 9.9], [2.7, 2.9, 2.9, 2.7], color='#D69E2E', lw=1.5)
    ax.text(7.45, 3.1, "T3: Nhóm C1 (Nhẹ)", ha='center', va='bottom', fontsize=8.5, weight='bold', color='#B7791F')

    # C2+
    ax.plot([10, 10, 27, 27], [2.7, 2.9, 2.9, 2.7], color='#C53030', lw=1.5)
    ax.text(18.5, 3.1, "T3: Nhóm C2+ (Nặng quy ước = C2 ∪ C3 ∪ C4)", ha='center', va='bottom', fontsize=8.5, weight='bold', color='#C53030')

    ax.set_xlim(-1, 28)
    ax.set_ylim(0.2, 3.6)
    ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig2_phq9_tiers.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 3. Figure 2: Gut-Brain & Metabolic Axis
# -------------------------------------------------------------
def make_fig2_gut_brain_metabolic_axis():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')

    # Central Node: Trầm cảm / Rối loạn khí sắc
    box_center = patches.FancyBboxPatch((0.36, 0.38), 0.28, 0.24, boxstyle="round,pad=0.03",
                                        fc='#FED7D7', ec='#C53030', lw=2)
    ax.add_patch(box_center)
    ax.text(0.50, 0.50, "TRIỆU CHỨNG TRẦM CẢM\n(PHQ-9 Severity)", ha='center', va='center',
            weight='bold', fontsize=11, color='#742A2A')

    # 4 Outer System Nodes:
    systems = [
        {"title": "PHẢN ỨNG VIÊM MÃN TÍNH\n& MIỄN DỊCH NGOẠI VI",
         "items": "• Albumin huyết thanh (LBXSAL)\n• Huyết sắc tố HbA1c (LBXGH)\n• Cytokine gây viêm (IL-6, TNF-α)\n• Rối loạn hàng rào máu-não",
         "box": (0.03, 0.58, 0.28, 0.36), "color": "#FEEBC8", "ec": "#C05621", "tc": "#7B341E",
         "arrow_start": (0.31, 0.70), "arrow_end": (0.38, 0.58)},
        
        {"title": "RỐI LOẠN CHUYỂN HÓA\n& LIPID HUYẾT TƯƠNG",
         "items": "• Chỉ số khối cơ thể (BMXBMI)\n• Chu vi vòng eo (BMXWAIST)\n• Cholesterol toàn phần (LBXTC)\n• HDL-Cholesterol (LBDHDD)",
         "box": (0.69, 0.58, 0.28, 0.36), "color": "#EBF8FF", "ec": "#2B6CB0", "tc": "#1A365D",
         "arrow_start": (0.69, 0.70), "arrow_end": (0.62, 0.58)},

        {"title": "HỆ THẦN KINH TỰ CHỦ\n& HUYẾT ĐỘNG HỌC",
         "items": "• Huyết áp tâm thu (SBP_mean)\n• Huyết áp tâm trương (DBP_mean)\n• Nhịp tim khi nghỉ (BPXPLS)\n• Rối loạn trương lực phó giao cảm",
         "box": (0.03, 0.06, 0.28, 0.36), "color": "#E6FFFA", "ec": "#2C7A7B", "tc": "#234E52",
         "arrow_start": (0.31, 0.30), "arrow_end": (0.38, 0.42)},

        {"title": "HÀNH VI LỐI SỐNG\n& NHỊP SINH HỌC",
         "items": "• Thời lượng giấc ngủ (SLD010H)\n• Vận động thể lực (PAQ605..)\n• Thời gian tĩnh tại (PAD680)\n• Hút thuốc & Sử dụng cồn",
         "box": (0.69, 0.06, 0.28, 0.36), "color": "#FAF5FF", "ec": "#6B46C1", "tc": "#322659",
         "arrow_start": (0.69, 0.30), "arrow_end": (0.62, 0.42)}
    ]

    for s in systems:
        x, y, w, h = s["box"]
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=s["color"], ec=s["ec"], lw=1.5)
        ax.add_patch(p)
        ax.text(x + w/2, y + h - 0.08, s["title"], ha='center', va='center', weight='bold', fontsize=9.5, color=s["tc"])
        ax.text(x + 0.02, y + 0.12, s["items"], ha='left', va='center', fontsize=8.2, color='#2D3748', linespacing=1.25)
        
        # Bi-directional arrows
        ax.annotate('', xy=s["arrow_end"], xytext=s["arrow_start"],
                    arrowprops=dict(facecolor=s["ec"], edgecolor=s["ec"], width=1.8, headwidth=6))
        ax.annotate('', xy=s["arrow_start"], xytext=s["arrow_end"],
                    arrowprops=dict(facecolor=s["ec"], edgecolor=s["ec"], width=1.8, headwidth=6))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig2_gut_brain_metabolic_axis.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 4. Figure 2: Class Imbalance & Feature Overlap Concept
# -------------------------------------------------------------
def make_fig2_imbalance_overlap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Subplot 1: Extreme Class Imbalance Bar
    classes = ['$C_0$\n(Tối thiểu)', '$C_1$\n(Nhẹ)', '$C_2$\n(Vừa)', '$C_3$\n(Nặng vừa)', '$C_4$\n(Rất nặng)']
    counts = [3016, 638, 243, 104, 44]
    pcts = [74.56, 15.77, 6.01, 2.57, 1.09]
    colors = ['#3182CE', '#ECC94B', '#DD6B20', '#E53E3E', '#822727']

    bars = ax1.bar(classes, counts, color=colors, edgecolor='#333333', lw=1)
    ax1.set_title('(a) Mất cân bằng lớp cực đoan (IR = 68.5:1)', fontsize=10.5, weight='bold', pad=12)
    ax1.set_ylabel('Số lượng mẫu quan sát ($N = 4.045$)', fontsize=9.5)
    ax1.set_ylim(0, 3500)
    for b, count, pct in zip(bars, counts, pcts):
        ax1.text(b.get_x() + b.get_width()/2, count + 80, f'{count}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=8.5, weight='bold')

    # Subplot 2: Conceptual Feature Distribution Overlap
    x = np.linspace(15, 45, 500)
    # C0 distribution
    y0 = 0.08 * np.exp(-0.5 * ((x - 28) / 5)**2)
    # C1 distribution
    y1 = 0.075 * np.exp(-0.5 * ((x - 29.5) / 5.5)**2)
    # C2+ distribution
    y2 = 0.065 * np.exp(-0.5 * ((x - 32) / 6)**2)

    ax2.plot(x, y0, label='$C_0$ (Tối thiểu, 74.6%)', color='#3182CE', lw=2)
    ax2.fill_between(x, y0, color='#3182CE', alpha=0.25)

    ax2.plot(x, y1, label='$C_1$ (Nhẹ, 15.8%)', color='#D69E2E', lw=2, linestyle='--')
    ax2.fill_between(x, y1, color='#ECC94B', alpha=0.25)

    ax2.plot(x, y2, label='$C_2^+$ (Nặng quy ước, 9.7%)', color='#E53E3E', lw=2, linestyle=':')
    ax2.fill_between(x, y2, color='#E53E3E', alpha=0.2)

    # Highlight Overlap Region
    overlap = np.minimum(y0, y1)
    ax2.fill_between(x, overlap, color='purple', alpha=0.4, label='Vùng chồng lấn Bayes (Chồng lấn cao)')

    ax2.set_title('(b) Chồng lấn phân phối đặc trưng sinh lý ($C_0$ vs $C_1$)', fontsize=10.5, weight='bold', pad=12)
    ax2.set_xlabel('Chỉ số sinh lý kết hợp (ví dụ: BMI / SBP z-score)', fontsize=9.5)
    ax2.set_ylabel('Mật độ xác suất $p(\\mathbf{x} \\mid y)$', fontsize=9.5)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig2_imbalance_overlap.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 5. Figure 2: Three Modeling Paradigms
# -------------------------------------------------------------
def make_fig2_modeling_paradigms():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5), dpi=300)

    for ax in (ax1, ax2, ax3):
        ax.axis('off')

    # 1. Flat Paradigm
    ax1.set_title('(a) Phân loại phẳng (Flat)', fontsize=10.5, weight='bold', color='#1A365D')
    p1 = patches.FancyBboxPatch((0.1, 0.7), 0.8, 0.2, boxstyle="round,pad=0.03", fc='#EBF8FF', ec='#2B6CB0', lw=1.5)
    ax1.add_patch(p1)
    ax1.text(0.5, 0.8, "Vectơ đặc trưng $\\mathbf{x} \\in \\mathbb{R}^d$", ha='center', va='center', weight='bold', fontsize=9.5)

    p_model = patches.FancyBboxPatch((0.2, 0.4), 0.6, 0.18, boxstyle="round,pad=0.03", fc='#ED8936', ec='#C05621', lw=1.5)
    ax1.add_patch(p_model)
    ax1.text(0.5, 0.49, "Bộ phân loại đa lớp\n(Softmax / Multi-tree)", ha='center', va='center', weight='bold', fontsize=9, color='white')

    ax1.annotate('', xy=(0.5, 0.58), xytext=(0.5, 0.7), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=5))
    ax1.annotate('', xy=(0.2, 0.18), xytext=(0.35, 0.4), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=4))
    ax1.annotate('', xy=(0.5, 0.18), xytext=(0.5, 0.4), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=4))
    ax1.annotate('', xy=(0.8, 0.18), xytext=(0.65, 0.4), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=4))

    ax1.text(0.2, 0.1, "$C_0$\n(Tối thiểu)", ha='center', fontsize=8.5, weight='bold', color='#2B6CB0')
    ax1.text(0.5, 0.1, "$C_1$\n(Nhẹ)", ha='center', fontsize=8.5, weight='bold', color='#D69E2E')
    ax1.text(0.8, 0.1, "$C_2^+$\n(Nặng)", ha='center', fontsize=8.5, weight='bold', color='#E53E3E')

    # 2. Hierarchical Paradigm
    ax2.set_title('(b) Phân loại phân cấp (Hierarchical)', fontsize=10.5, weight='bold', color='#234E52')
    p2 = patches.FancyBboxPatch((0.1, 0.82), 0.8, 0.14, boxstyle="round,pad=0.03", fc='#E6FFFA', ec='#2C7A7B', lw=1.5)
    ax2.add_patch(p2)
    ax2.text(0.5, 0.89, "Đặc trưng $\\mathbf{x}$", ha='center', va='center', weight='bold', fontsize=9.5)

    p_stage1 = patches.FancyBboxPatch((0.2, 0.58), 0.6, 0.14, boxstyle="round,pad=0.03", fc='#319795', ec='#234E52', lw=1.5)
    ax2.add_patch(p_stage1)
    ax2.text(0.5, 0.65, "Giai đoạn 1: $M_1$\nSàng lọc ($C_0$ vs $C_{12}$)", ha='center', va='center', weight='bold', fontsize=8.5, color='white')

    p_stage2 = patches.FancyBboxPatch((0.45, 0.32), 0.5, 0.14, boxstyle="round,pad=0.03", fc='#319795', ec='#234E52', lw=1.5)
    ax2.add_patch(p_stage2)
    ax2.text(0.7, 0.39, "Giai đoạn 2: $M_2$\nPhân tầng ($C_1$ vs $C_2^+$)", ha='center', va='center', weight='bold', fontsize=8.5, color='white')

    ax2.annotate('', xy=(0.5, 0.72), xytext=(0.5, 0.82), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=5))
    ax2.annotate('', xy=(0.25, 0.35), xytext=(0.35, 0.58), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=5))
    ax2.annotate('', xy=(0.7, 0.46), xytext=(0.65, 0.58), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=5))
    ax2.annotate('', xy=(0.55, 0.12), xytext=(0.65, 0.32), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=4))
    ax2.annotate('', xy=(0.85, 0.12), xytext=(0.75, 0.32), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=4))

    ax2.text(0.25, 0.28, "$C_0$ (Loại bỏ)\n[Rủi ro lan truyền]", ha='center', fontsize=8, weight='bold', color='#C53030')
    ax2.text(0.55, 0.05, "$C_1$ (Nhẹ)", ha='center', fontsize=8, weight='bold', color='#D69E2E')
    ax2.text(0.85, 0.05, "$C_2^+$ (Nặng)", ha='center', fontsize=8, weight='bold', color='#2B6CB0')

    # 3. Ordinal Paradigm
    ax3.set_title('(c) Phân loại thứ bậc (Ordinal)', fontsize=10.5, weight='bold', color='#553C9A')
    p3 = patches.FancyBboxPatch((0.1, 0.82), 0.8, 0.14, boxstyle="round,pad=0.03", fc='#FAF5FF', ec='#6B46C1', lw=1.5)
    ax3.add_patch(p3)
    ax3.text(0.5, 0.89, "Đặc trưng $\\mathbf{x}$", ha='center', va='center', weight='bold', fontsize=9.5)

    p_ord = patches.FancyBboxPatch((0.15, 0.48), 0.7, 0.22, boxstyle="round,pad=0.03", fc='#805AD5', ec='#553C9A', lw=1.5)
    ax3.add_patch(p_ord)
    ax3.text(0.5, 0.59, "Hàm tiềm ẩn $z = \\mathbf{w}^T \\mathbf{x}$\n& Các ngưỡng cắt $\\theta_1 < \\theta_2$",
             ha='center', va='center', weight='bold', fontsize=8.5, color='white')

    ax3.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.82), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=5))
    ax3.annotate('', xy=(0.2, 0.25), xytext=(0.3, 0.48), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=4))
    ax3.annotate('', xy=(0.5, 0.25), xytext=(0.5, 0.48), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=4))
    ax3.annotate('', xy=(0.8, 0.25), xytext=(0.7, 0.48), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=4))

    ax3.text(0.2, 0.16, "$z < \\theta_1$\n$\\to C_0$", ha='center', fontsize=8.5, weight='bold', color='#2B6CB0')
    ax3.text(0.5, 0.16, "$\\theta_1 \\leq z < \\theta_2$\n$\\to C_1$", ha='center', fontsize=8.5, weight='bold', color='#D69E2E')
    ax3.text(0.8, 0.16, "$z \\geq \\theta_2$\n$\\to C_2^+$", ha='center', fontsize=8.5, weight='bold', color='#E53E3E')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig2_modeling_paradigms.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 6. Figure 3.2: Data Pipeline Flowchart
# -------------------------------------------------------------
def make_fig3_2_data_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    ax.axis('off')

    steps = [
        {"title": "1. NHẬN DỮ LIỆU GỐC", "desc": "• File SAS Transport (.XPT)\n• 3 chu kỳ NHANES (11-16)\n• Khám lâm sàng & Xét nghiệm", "box": (0.02, 0.35, 0.18, 0.45), "color": "#EBF8FF", "ec": "#2B6CB0"},
        {"title": "2. LỌC ĐỐI TƯỢNG", "desc": "• Người lớn $\\geq$ 20 tuổi\n• Hợp lệ 9 câu hỏi DPQ\n• Loại bỏ đối tượng khuyết\n  hoàn toàn khảo sát", "box": (0.22, 0.35, 0.18, 0.45), "color": "#E6FFFA", "ec": "#2C7A7B"},
        {"title": "3. HÀI HÒA HÓA", "desc": "• Ghép nối qua khóa SEQN\n• Lọc biến khuyết cấu trúc\n• Thống nhất schema 31 biến\n  ($F_{4\\_\\text{common31}}$)", "box": (0.42, 0.35, 0.18, 0.45), "color": "#FEFCBF", "ec": "#B7791F"},
        {"title": "4. CÔ LẬP N-FOLD", "desc": "• Stratified 5-Fold Split\n• Băm bảo mật SHA-256\n• Khóa phân hoạch chuẩn\n  (Zero Data Leakage)", "box": (0.62, 0.35, 0.18, 0.45), "color": "#FEEBC8", "ec": "#C05621"},
        {"title": "5. TIỀN XỬ LÝ IN-FOLD", "desc": "• Imputer (fit trên train)\n• Scaler (fit trên train)\n• OneHotEncoder (fit train)\n• Áp dụng sang val/test", "box": (0.82, 0.35, 0.17, 0.45), "color": "#FED7D7", "ec": "#C53030"}
    ]

    for i, s in enumerate(steps):
        x, y, w, h = s["box"]
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=s["color"], ec=s["ec"], lw=1.5)
        ax.add_patch(p)
        ax.text(x + w/2, y + h - 0.08, s["title"], ha='center', va='center', weight='bold', fontsize=8.5, color='#1A202C')
        ax.text(x + 0.015, y + 0.15, s["desc"], ha='left', va='center', fontsize=8, color='#2D3748', linespacing=1.2)

        if i < len(steps) - 1:
            ax.annotate('', xy=(x + w + 0.015, 0.57), xytext=(x + w - 0.005, 0.57),
                        arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=1.5, headwidth=5))

    # Bottom Banner: Zero Leakage Guarantee
    p_banner = patches.FancyBboxPatch((0.02, 0.08), 0.97, 0.18, boxstyle="round,pad=0.03", fc='#EDF2F7', ec='#4A5568', lw=1.2)
    ax.add_patch(p_banner)
    ax.text(0.5, 0.17, "NGUYÊN TẮC BẢO TOÀN DỮ LIỆU: ZERO DATA LEAKAGE", ha='center', va='center', weight='bold', fontsize=9.5, color='#2D3748')
    ax.text(0.5, 0.11, "Không đưa bất kỳ biến câu hỏi DPQ vào ma trận đầu vào. Tất cả các tham số trung vị (median) và độ lệch chuẩn đều học độc lập trong từng fold huấn luyện.",
            ha='center', va='center', fontsize=8.2, color='#4A5568')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig3_2_data_pipeline.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 7. Figure 3.3: Feature Group Hierarchy (F1 -> F4)
# -------------------------------------------------------------
def make_fig3_3_feature_groups():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.axis('off')

    groups = [
        {"name": "Nhóm F1 (10 biến cốt lõi)", "sub": "Thể chất, Sinh tồn & Xét nghiệm máu",
         "vars": "• Thể chất & Huyết động: BMXBMI, BMXWAIST, BPXPLS, SBP_mean, DBP_mean\n• Sinh hóa máu: HbA1c (LBXGH), Cholesterol (LBXTC), HDL (LBDHDD), Creatinine (LBXSCR), Albumin (LBXSAL)",
         "box": (0.02, 0.72, 0.96, 0.22), "color": "#EBF8FF", "ec": "#3182CE"},
        
        {"name": "Nhóm F2 (19 biến = F1 + Tiền sử bệnh lâm sàng)", "sub": "Bổ sung 9 biến bệnh mạn tính",
         "vars": "• Bổ sung: Tăng huyết áp (BPQ020), Đái tháo đường (DIQ010), 5 bệnh tim mạch & đột quỵ (MCQ160B, C, D, E, F),\n  Ung thư (MCQ220), Tự đánh giá sức khỏe tổng quát (HUQ010)",
         "box": (0.02, 0.48, 0.96, 0.20), "color": "#E6FFFA", "ec": "#319795"},

        {"name": "Nhóm F3 (29 biến = F2 + Hành vi lối sống & Giấc ngủ)", "sub": "Bổ sung 10 biến thói quen sinh hoạt",
         "vars": "• Bổ sung: Hoạt động thể lực (PAQ605, PAQ620, PAQ650, PAQ665), Thời gian tĩnh tại (PAD680),\n  Tình trạng hút thuốc (smoking_status), Thói quen uống rượu (ALQ101, alcohol_status, alcohol_quantity_code), Thời lượng ngủ (SLD010H)",
         "box": (0.02, 0.25, 0.96, 0.19), "color": "#FEFCBF", "ec": "#D69E2E"},

        {"name": "Nhóm F4 (34 biến = F3 + Nhân khẩu học & Kinh tế xã hội)", "sub": "Bổ sung 5 biến nhân khẩu",
         "vars": "• Bổ sung: Tuổi (RIDAGEYR), Giới tính (RIAGENDR), Chủng tộc (RIDRETH1), Trình độ học vấn (DMDEDUC2), Tỷ lệ nghèo gia đình (INDFMPIR)",
         "box": (0.02, 0.03, 0.96, 0.18), "color": "#FED7D7", "ec": "#E53E3E"}
    ]

    for g in groups:
        x, y, w, h = g["box"]
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=g["color"], ec=g["ec"], lw=1.5)
        ax.add_patch(p)
        ax.text(x + 0.02, y + h - 0.05, f"{g['name']}  --  {g['sub']}", weight='bold', fontsize=9, color='#1A202C')
        ax.text(x + 0.03, y + 0.06, g["vars"], fontsize=8, color='#2D3748', linespacing=1.2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig3_3_feature_groups.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 8. Figure 3.4: Hard Hierarchical Architecture & Error Propagation
# -------------------------------------------------------------
def make_fig3_4_hard_hierarchical():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=300)
    ax.axis('off')

    # Input
    p_in = patches.FancyBboxPatch((0.02, 0.42), 0.18, 0.22, boxstyle="round,pad=0.03", fc='#EBF8FF', ec='#2B6CB0', lw=1.5)
    ax.add_patch(p_in)
    ax.text(0.11, 0.53, "Vectơ đầu vào $\\mathbf{x}$\n(Tập dữ liệu $N = 4.045$)", ha='center', va='center', weight='bold', fontsize=9)

    # Arrow 1 -> M1
    ax.annotate('', xy=(0.28, 0.53), xytext=(0.20, 0.53), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=6))

    # Stage 1 Model: M1
    p_m1 = patches.FancyBboxPatch((0.28, 0.38), 0.26, 0.30, boxstyle="round,pad=0.03", fc='#319795', ec='#234E52', lw=2)
    ax.add_patch(p_m1)
    ax.text(0.41, 0.57, "GIAI ĐOẠN 1: MÔ HÌNH $M_1$", ha='center', va='center', weight='bold', fontsize=9.5, color='white')
    ax.text(0.41, 0.47, "Phân loại nhị phân:\n$C_0$ (Tối thiểu) vs $C_{12}$ (Có triệu chứng)", ha='center', va='center', fontsize=8.2, color='#E6FFFA')

    # Branch M1 -> C0 (Hard Decision)
    ax.annotate('', xy=(0.60, 0.72), xytext=(0.54, 0.58), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=6))
    p_c0 = patches.FancyBboxPatch((0.60, 0.64), 0.36, 0.24, boxstyle="round,pad=0.03", fc='#FED7D7', ec='#C53030', lw=1.8)
    ax.add_patch(p_c0)
    ax.text(0.78, 0.78, "Dự đoán lớp $C_0$ (Loại bỏ khỏi luồng)", weight='bold', fontsize=9, color='#742A2A', ha='center')
    ax.text(0.78, 0.70, "[ĐIỂM NGHẼN LAN TRUYỀN SAI SỐ]", weight='bold', fontsize=8.5, color='#9B2C2C', ha='center')
    ax.text(0.78, 0.65, "Bỏ sót 120 ca $C_2^+$ (E1) do quyết định cứng cắt đứt cơ hội!", fontsize=7.8, color='#742A2A', ha='center')

    # Branch M1 -> M2
    ax.annotate('', xy=(0.60, 0.34), xytext=(0.54, 0.48), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=6))
    
    # Stage 2 Model: M2
    p_m2 = patches.FancyBboxPatch((0.60, 0.20), 0.22, 0.26, boxstyle="round,pad=0.03", fc='#2B6CB0', ec='#1A365D', lw=2)
    ax.add_patch(p_m2)
    ax.text(0.71, 0.36, "GIAI ĐOẠN 2: $M_2$", ha='center', va='center', weight='bold', fontsize=9.5, color='white')
    ax.text(0.71, 0.27, "Chỉ đánh giá các mẫu\nđược $M_1$ dự đoán $C_{12}$:\n$C_1$ (Nhẹ) vs $C_2^+$ (Nặng)", ha='center', va='center', fontsize=8, color='#EBF8FF')

    # Outputs of M2
    ax.annotate('', xy=(0.88, 0.38), xytext=(0.82, 0.34), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=5))
    ax.annotate('', xy=(0.88, 0.20), xytext=(0.82, 0.26), arrowprops=dict(facecolor='#4A5568', width=1.2, headwidth=5))

    p_out_c1 = patches.FancyBboxPatch((0.88, 0.32), 0.10, 0.12, boxstyle="round,pad=0.02", fc='#FEFCBF', ec='#D69E2E', lw=1.2)
    ax.add_patch(p_out_c1)
    ax.text(0.93, 0.38, "Lớp $C_1$\n(Nhẹ)", ha='center', va='center', weight='bold', fontsize=8.5, color='#744210')

    p_out_c2 = patches.FancyBboxPatch((0.88, 0.14), 0.10, 0.12, boxstyle="round,pad=0.02", fc='#C6F6D5', ec='#276749', lw=1.2)
    ax.add_patch(p_out_c2)
    ax.text(0.93, 0.20, "Lớp $C_2^+$\n(Nặng)", ha='center', va='center', weight='bold', fontsize=8.5, color='#22543D')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig3_4_hard_hierarchical.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 9. Figure 3.5: Soft Cost-Sensitive Architecture & Bayes Matrix
# -------------------------------------------------------------
def make_fig3_5_soft_cost_sensitive():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=300)
    ax.axis('off')

    # Left: Soft Fusion Probability
    p1 = patches.FancyBboxPatch((0.02, 0.20), 0.42, 0.70, boxstyle="round,pad=0.03", fc='#E6FFFA', ec='#2C7A7B', lw=1.8)
    ax.add_patch(p1)
    ax.text(0.23, 0.83, "KẾT HỢP XÁC SUẤT MỀM (SOFT FUSION)", weight='bold', fontsize=9.5, color='#234E52', ha='center')
    
    soft_eqs = [
        "1. Xác suất Stage 1 ($M_1$):",
        "   $P(C_0 \\mid \\mathbf{x}) = 1 - P(C_{12} \\mid \\mathbf{x})$",
        "   $P(C_{12} \\mid \\mathbf{x})$ là xác suất có triệu chứng.",
        "",
        "2. Xác suất Stage 2 ($M_2$):",
        "   $P(C_1 \\mid C_{12}, \\mathbf{x})$ và $P(C_2^+ \\mid C_{12}, \\mathbf{x})$",
        "",
        "3. Tích hợp phân phối hậu nghiệm đầy đủ:",
        "   $\\hat{P}(C_0 \\mid \\mathbf{x}) = P(C_0 \\mid \\mathbf{x})$",
        "   $\\hat{P}(C_1 \\mid \\mathbf{x}) = P(C_{12} \\mid \\mathbf{x}) \\cdot P(C_1 \\mid C_{12}, \\mathbf{x})$",
        "   $\\hat{P}(C_2^+ \\mid \\mathbf{x}) = P(C_{12} \\mid \\mathbf{x}) \\cdot P(C_2^+ \\mid C_{12}, \\mathbf{x})$",
        "   => Tổng xác suất = 1 (Bảo toàn độ đo xác suất!)"
    ]
    ax.text(0.04, 0.49, "\n".join(soft_eqs), fontsize=8.2, color='#1A202C', linespacing=1.2)

    # Arrow -> Right
    ax.annotate('', xy=(0.50, 0.55), xytext=(0.45, 0.55), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=6))

    # Right: Bayes Minimum Risk Decision
    p2 = patches.FancyBboxPatch((0.50, 0.15), 0.48, 0.75, boxstyle="round,pad=0.03", fc='#FEFCBF', ec='#B7791F', lw=1.8)
    ax.add_patch(p2)
    ax.text(0.74, 0.83, "QUYẾT ĐỊNH TỐI THIỂU HÓA RỦI RO BAYES", weight='bold', fontsize=9.5, color='#744210', ha='center')

    bayes_text = [
        "Quy tắc quyết định Bayes:",
        "$\\hat{y}^* = \\arg\\min_i \\sum_{j=0}^2 \\mathbf{C}(i, j) \\hat{P}(C_j \\mid \\mathbf{x})$",
        "",
        "Ma trận chi phí phạt lỗi nặng (M3):",
        "  C(0,2) = 5 (Phạt nặng nhất: bỏ sót ca nặng C2+ -> C0)",
        "  C(1,2) = 2 (Phạt nhầm nặng sang nhẹ)",
        "  C(2,0) = 1 (Phạt báo động nhầm)",
        "",
        "Kết quả đánh đổi lâm sàng:",
        "• Giảm bỏ sót từ 120 ca xuống còn 62 ca (E5)",
        "  và chỉ còn 31 ca (E6b)!",
        "• Đổi lại: Báo động nhầm tăng từ 432 lên 1.343 ca."
    ]
    ax.text(0.52, 0.46, "\n".join(bayes_text), fontsize=8.2, color='#744210', linespacing=1.2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig3_5_soft_cost_sensitive.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 10. Figure 3.6: Ordinal Classification Principles
# -------------------------------------------------------------
def make_fig3_6_ordinal_classification():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Subplot 1: Latent Variable & Threshold Hyperplanes (OLR)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-0.1, 1.1)
    z = np.linspace(-4, 4, 300)
    sig = 1 / (1 + np.exp(-z))
    ax1.plot(z, sig, color='#2B6CB0', lw=2, label='$\\sigma(z) = \\frac{1}{1 + e^{-z}}$')
    
    # Thresholds
    theta1 = -0.8
    theta2 = 1.2
    ax1.axvline(theta1, color='#C53030', linestyle='--', lw=1.5, label='Ngưỡng $\\theta_1$ (Ranh giới $C_0 \\mid C_1$)')
    ax1.axvline(theta2, color='#C53030', linestyle=':', lw=1.5, label='Ngưỡng $\\theta_2$ (Ranh giới $C_1 \\mid C_2^+$)')

    ax1.text((theta1 - 4)/2, 0.5, 'Vùng $C_0$\n($s < \\theta_1$)', ha='center', fontsize=9, weight='bold', color='#2B6CB0')
    ax1.text((theta1 + theta2)/2, 0.5, 'Vùng $C_1$\n($\\theta_1 \\leq s < \\theta_2$)', ha='center', fontsize=9, weight='bold', color='#D69E2E')
    ax1.text((theta2 + 4)/2, 0.5, 'Vùng $C_2^+$\n($s \\geq \\theta_2$)', ha='center', fontsize=9, weight='bold', color='#E53E3E')

    ax1.set_title('(a) Hồi quy Logistic thứ bậc (OLR)\nBiểu diễn biến tiềm ẩn và ngưỡng cắt song song', fontsize=10, weight='bold', pad=10)
    ax1.set_xlabel('Hàm tiềm ẩn $z = \\mathbf{w}^T \\mathbf{x}$', fontsize=9)
    ax1.set_ylabel('Xác suất tích lũy $P(Y \\leq k \\mid \\mathbf{x})$', fontsize=9)
    ax1.legend(loc='upper left', fontsize=8)

    # Subplot 2: Cumulative Binary Classification (CBC) & Violation
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1.1)

    # Sample cases showing monotonic vs non-monotonic
    cases = ['Mẫu A\n(Chuẩn)', 'Mẫu B\n(Chuẩn)', 'Mẫu C\n(Vi phạm\nđơn điệu!)', 'Mẫu D\n(Chuẩn)']
    x_pos = [1.5, 4.0, 6.5, 9.0]
    p_k1 = [0.85, 0.65, 0.30, 0.90]  # P(Y >= 1)
    p_k2 = [0.20, 0.15, 0.45, 0.10]  # P(Y >= 2)

    width = 0.6
    ax2.bar([p - width/2 for p in x_pos], p_k1, width=width, label='$P(Y \\geq 1 \\mid \\mathbf{x})$ (Bộ phân loại 1)', color='#3182CE', alpha=0.85)
    ax2.bar([p + width/2 for p in x_pos], p_k2, width=width, label='$P(Y \\geq 2 \\mid \\mathbf{x})$ (Bộ phân loại 2)', color='#E53E3E', alpha=0.85)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cases, fontsize=8.5, weight='bold')
    ax2.set_title('(b) Phân loại nhị phân tích lũy (CBC)\nNguy cơ vi phạm tính đơn điệu ($P_{k+1} > P_k$)', fontsize=10, weight='bold', pad=10)
    ax2.set_ylabel('Xác suất dự đoán từ 2 bộ phân loại độc lập', fontsize=9)
    ax2.legend(loc='upper right', fontsize=8)

    # Annotation on violation
    ax2.annotate('Vi phạm: $P(Y \\geq 2) > P(Y \\geq 1)$\n(Tỷ lệ 0.32% trong E7b)',
                 xy=(6.5, 0.46), xytext=(4.5, 0.8),
                 arrowprops=dict(facecolor='black', width=1, headwidth=5),
                 fontsize=8, weight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#FED7D7', ec='#C53030'))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig3_6_ordinal_classification.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# 11. Figure 3.7: Nested CV & Temporal Protocol
# -------------------------------------------------------------
def make_fig3_7_evaluation_protocol():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=300)
    ax.axis('off')

    # Top: Nested 5-Fold CV on Development Cohort
    p_top = patches.FancyBboxPatch((0.02, 0.52), 0.96, 0.42, boxstyle="round,pad=0.03", fc='#EBF8FF', ec='#2B6CB0', lw=1.8)
    ax.add_patch(p_top)
    ax.text(0.04, 0.88, "QUY TRÌNH KIỂM ĐỊNH LỒNG NHAU (NESTED 5-FOLD STRATIFIED CV)", weight='bold', fontsize=9.5, color='#1A365D')
    
    cv_desc = [
        "Tập phát triển Legacy (NHANES 2013-2014, N = 4.045 đối tượng):",
        "• Outer CV (5 folds): Phân hoạch ngẫu nhiên tầng hóa theo nhãn mục tiêu (giữ nguyên tỷ lệ 74.6% : 15.8% : 9.7%).",
        "• Inner CV (3 folds trên từng Outer Training set): Tinh chỉnh siêu tham số (RandomizedSearchCV) tối ưu Macro-F1.",
        "• In-fold Preprocessing: Toàn bộ khâu điền khuyết (median/mode) và chuẩn hóa (scaling) fit 100% bên trong từng fold huấn luyện.",
        "• Đánh giá OOF (Out-Of-Fold): Thu nhận dự đoán trên fold kiểm tra ngoài, tính trung bình và độ lệch chuẩn (Mean ± Std)."
    ]
    ax.text(0.04, 0.68, "\n".join(cv_desc), fontsize=8.2, color='#2D3748', linespacing=1.2)

    # Arrow Down
    ax.annotate('', xy=(0.50, 0.46), xytext=(0.50, 0.52), arrowprops=dict(facecolor='#4A5568', width=1.5, headwidth=6))

    # Bottom: Temporal Out-of-cycle Validation
    p_bot = patches.FancyBboxPatch((0.02, 0.04), 0.96, 0.40, boxstyle="round,pad=0.03", fc='#E6FFFA', ec='#2C7A7B', lw=1.8)
    ax.add_patch(p_bot)
    ax.text(0.04, 0.38, "KIỂM ĐỊNH ĐỘC LẬP THEO DÒNG THỜI GIAN (TEMPORAL OUT-OF-CYCLE VALIDATION)", weight='bold', fontsize=9.5, color='#234E52')

    temporal_desc = [
        "Tập huấn luyện (Training Cohort): NHANES 2013-2014 (N = 5.054, Branch A)",
        "Tập kiểm tra tương lai độc lập (Temporal Test Cohort): NHANES 2015-2016 (N = 4.888, Branch D)",
        "• Kiểm tra độ trôi phân phối y sinh (biomedical covariate shift) theo thời gian.",
        "• Hoàn toàn tách biệt về mặt thời gian và đối tượng khảo sát.",
        "• Chứng minh tính tổng quát hóa thực tế của mô hình: OLR đạt BalAcc = 0.5073 (so với In-cycle CV = 0.5151)."
    ]
    ax.text(0.04, 0.18, "\n".join(temporal_desc), fontsize=8.2, color='#2D3748', linespacing=1.2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig3_7_evaluation_protocol.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# Run all generator functions
if __name__ == '__main__':
    print("Generating complete scientific figures suite...")
    make_fig1_problem_formulation()
    make_fig2_phq9_tiers()
    make_fig2_gut_brain_metabolic_axis()
    make_fig2_imbalance_overlap()
    make_fig2_modeling_paradigms()
    make_fig3_2_data_pipeline()
    make_fig3_3_feature_groups()
    make_fig3_4_hard_hierarchical()
    make_fig3_5_soft_cost_sensitive()
    make_fig3_6_ordinal_classification()
    make_fig3_7_evaluation_protocol()
    print("All scientific figures successfully generated!")
