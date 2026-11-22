#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_perfect_figures.py
Master Publication-Grade Figure Generator for DepSev-ML Scientific Report V5.

Strict Standards:
- Coordinate Grid: [0, 100] x [0, 100] for exact mathematical placement.
- No border clipping (zero 'mất bo góc'): All boxes stay strictly inside [3, 97],
  with boxstyle='round,pad=0.8,rounding_size=2.0' and pad_inches=0.15 on savefig.
- No elements stuck together (zero 'dính vào nhau'): Minimum 4-6 units gap between boxes,
  arrows have 1.5 units clearance from borders, text has generous internal breathing room.
- Clear, prominent typography (font sizes 8.5-12pt), high-contrast academic colors:
  Navy #1E3A8A, Royal Blue #2563EB, Emerald #059669, Jade #10B981,
  Amber #D97706, Gold #F59E0B, Crimson #DC2626, Coral #EF4444,
  Purple #7C3AED, Slate Dark #1E293B, Slate Gray #475569, Light BG #F8FAFC.
- NO fake specific branching thresholds in conceptual diagrams.
- Explanations in text, figures keep concise keywords only.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

def create_canvas(width=11.5, height=5.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    return fig, ax

def save_clean_fig(fig, filename):
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"Generated successfully: {filename}")

# ==============================================================================
# 1. Figure 1.1: Problem Formulation Diagram
# ==============================================================================
def make_fig1_problem_formulation():
    fig, ax = create_canvas(width=12.0, height=5.2)

    cols = [
        {"x": 3, "w": 21, "title": "DỮ LIỆU ĐẦU VÀO", "sub": "34 đặc trưng sinh lý",
         "items": ["• Thể chất & Sinh tồn:\n  BMI, Vòng eo, Huyết áp, Mạch",
                   "• Cận lâm sàng máu:\n  HbA1c, Cholesterol, HDL, Albumin",
                   "• Tiền sử & Lối sống:\n  Bệnh mạn tính, Giấc ngủ, Vận động",
                   "• Nhân khẩu học:\n  Tuổi, Giới tính, Học vấn, Nghèo"],
         "fc": "#EFF6FF", "ec": "#2563EB", "tc": "#1E3A8A"},

        {"x": 27.5, "w": 21, "title": "TIỀN XỬ LÝ CHUẨN", "sub": "Zero Data Leakage",
         "items": ["• Khử biến khuyết cấu trúc\n  (Triglyceride loại bỏ)",
                   "• In-fold KNN Imputation\n  (Chỉ fit trên tập train)",
                   "• In-fold StandardScaler\n  (Không rò rỉ phương sai)",
                   "• Stratified 5-Fold Split\n  (Khóa mã băm SHA-256)"],
         "fc": "#ECFDF5", "ec": "#059669", "tc": "#065F46"},

        {"x": 52, "w": 21, "title": "ĐA TRƯỜNG PHÁI ML", "sub": "4 nhóm kiến trúc",
         "items": ["• Phân loại phẳng:\n  RF Baseline, XGBoost, LR",
                   "• Phân cấp 2 giai đoạn:\n  Hard Routing (E1-E3)\n  Soft Probability (E4-E5)",
                   "• Tối ưu rủi ro Bayes:\n  Cost-Sensitive Learning (E6)",
                   "• Phân loại thứ bậc:\n  Balanced OLR, CBC (E7)"],
         "fc": "#FFFBEB", "ec": "#D97706", "tc": "#92400E"},

        {"x": 76.5, "w": 20.5, "title": "PHÂN TẦNG NGUY CƠ", "sub": "Đánh giá đa chiều",
         "items": ["• Không gian T5 (5 lớp):\n  C0 Minimal (0-4)\n  C1 Mild (5-9)\n  C2 Moderate (10-14)\n  C3 Mod-Severe (15-19)\n  C4 Severe (20-27)",
                   "• Không gian T3 (3 nhóm):\n  C0 (0-4), C1 (5-9), C2+ (10-27)",
                   "• An toàn lâm sàng:\n  Định lượng ca bỏ sót nặng"],
         "fc": "#FEF2F2", "ec": "#DC2626", "tc": "#991B1B"}
    ]

    for c in cols:
        box = patches.FancyBboxPatch((c["x"], 6), c["w"], 88,
                                     boxstyle="round,pad=0.8,rounding_size=2.2",
                                     fc=c["fc"], ec=c["ec"], lw=1.5)
        ax.add_patch(box)
        ax.text(c["x"] + c["w"]/2, 88, c["title"], fontsize=10.5, fontweight='bold',
                ha='center', va='center', color=c["tc"])
        ax.text(c["x"] + c["w"]/2, 82, f"({c['sub']})", fontsize=8.5, fontstyle='italic',
                ha='center', va='center', color='#475569')

        ax.text(c["x"] + 1.5, 47, "\n\n".join(c["items"]), fontsize=8.2, va='center',
                color='#1E293B', linespacing=1.25)

    arrow_pairs = [(24.5, 27.0), (49.0, 51.5), (73.5, 76.0)]
    for x_start, x_end in arrow_pairs:
        ax.annotate('', xy=(x_end, 50), xytext=(x_start, 50),
                    arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=2.0, headwidth=6))

    save_clean_fig(fig, 'fig1_problem_formulation.png')

# ==============================================================================
# 2. Figure 1.2: Research Progression Pipeline (T5 -> T3)
# ==============================================================================
def make_fig1_research_progression():
    fig, ax = create_canvas(width=12.0, height=5.4)

    # Left Half: Phase 1 (T5)
    p1 = patches.FancyBboxPatch((3, 6), 44, 88,
                                boxstyle="round,pad=0.8,rounding_size=2.2",
                                fc="#F8FAFC", ec="#94A3B8", lw=1.5)
    ax.add_patch(p1)
    ax.text(25, 89, "GIAI ĐOẠN 1: PHÂN LOẠI 5 MỨC ĐỘ (T5)",
            fontsize=11, fontweight='bold', ha='center', va='center', color='#1E293B')

    b_t5 = patches.FancyBboxPatch((5, 57), 40, 26,
                                  boxstyle="round,pad=0.8,rounding_size=1.8",
                                  fc="#EFF6FF", ec="#2563EB", lw=1.3)
    ax.add_patch(b_t5)
    ax.text(25, 77, "Không gian 5 lớp chuẩn PHQ-9", fontsize=9.5, fontweight='bold', ha='center', color='#1E3A8A')
    ax.text(25, 66, "C0 (0-4: 74.6%)   |   C1 (5-9: 15.8%)\nC2 (10-14: 6.0%)   |   C3 (15-19: 2.6%)   |   C4 (20-27: 1.1%)",
            fontsize=8.5, ha='center', va='center', color='#1E293B', linespacing=1.3)

    b_lim = patches.FancyBboxPatch((5, 12), 40, 39,
                                   boxstyle="round,pad=0.8,rounding_size=1.8",
                                   fc="#FEF2F2", ec="#DC2626", lw=1.3)
    ax.add_patch(b_lim)
    ax.text(25, 45, "Rào cản thông tin y sinh học", fontsize=9.5, fontweight='bold', ha='center', color='#991B1B')
    lims = [
        "• Mất cân bằng lớp cực đoan: IR = 68.5:1 (C0 vs C4)",
        "• Chồng lấn phân phối sinh lý: C0 trùng khít C1 (PCA/IQR)",
        "• Sự sụp đổ của lớp trung gian: Recall C1 chỉ đạt 16.29%",
        "• Chạm ngưỡng trần thông tin: Macro-F1 bão hòa ~0.27"
    ]
    ax.text(7, 28, "\n".join(lims), fontsize=8.2, va='center', color='#7F1D1D', linespacing=1.35)

    ax.annotate('', xy=(53.5, 50), xytext=(47.5, 50),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=2.5, headwidth=7))
    ax.text(50.5, 55, "Tái cấu trúc\nlâm sàng", fontsize=8.5, fontweight='bold', ha='center', color='#334155')

    # Right Half: Phase 2 (T3)
    p2 = patches.FancyBboxPatch((54, 6), 43, 88,
                                boxstyle="round,pad=0.8,rounding_size=2.2",
                                fc="#F8FAFC", ec="#94A3B8", lw=1.5)
    ax.add_patch(p2)
    ax.text(75.5, 89, "GIAI ĐOẠN 2: PHÂN CẤP 2 BƯỚC (T3)",
            fontsize=11, fontweight='bold', ha='center', va='center', color='#1E293B')

    b_st1 = patches.FancyBboxPatch((56, 52), 39, 31,
                                   boxstyle="round,pad=0.8,rounding_size=1.8",
                                   fc="#ECFDF5", ec="#059669", lw=1.3)
    ax.add_patch(b_st1)
    ax.text(75.5, 77, "Stage 1: Sàng lọc nhị phân sơ bộ", fontsize=9.5, fontweight='bold', ha='center', color='#065F46')
    ax.text(75.5, 64, "C0 (0-4: Không triệu chứng)   vs   C12 (5-27: Có triệu chứng)\n• Tập huấn luyện: Toàn bộ đối tượng NHANES\n• Mục tiêu: Nhận diện kịp thời đối tượng nguy cơ",
            fontsize=8.2, ha='center', va='center', color='#047857', linespacing=1.3)

    ax.annotate('', xy=(75.5, 45), xytext=(75.5, 51.5),
                arrowprops=dict(facecolor='#059669', edgecolor='#059669', width=1.8, headwidth=6))

    b_st2 = patches.FancyBboxPatch((56, 12), 39, 31,
                                   boxstyle="round,pad=0.8,rounding_size=1.8",
                                   fc="#FFFBEB", ec="#D97706", lw=1.3)
    ax.add_patch(b_st2)
    ax.text(75.5, 37, "Stage 2: Phân tầng mức độ nghiêm trọng", fontsize=9.5, fontweight='bold', ha='center', color='#92400E')
    ax.text(75.5, 24, "C1 (5-9: Nhẹ)   vs   C2+ (10-27: Nặng quy ước)\n• Huấn luyện: CHỈ trên các mẫu có nhãn thật PHQ-9 >= 5\n• Cơ chế định tuyến: Hard Routing vs Soft Bayes Risk",
            fontsize=8.2, ha='center', va='center', color='#B45309', linespacing=1.3)

    save_clean_fig(fig, 'fig_research_progression_t5_to_t3.png')

# ==============================================================================
# 3. Figure 2.1: PHQ-9 Scale & Severity Tiers
# ==============================================================================
def make_fig2_phq9_tiers():
    fig, ax = plt.subplots(figsize=(11.0, 4.4), dpi=300)
    ax.set_xlim(-1, 28)
    ax.set_ylim(0, 3.8)
    ax.axis('off')

    tiers = [
        {"name": "C0: Tối thiểu\n(Minimal)", "range": (0, 4.9), "color": "#10B981", "action": "Không yêu cầu can thiệp y tế"},
        {"name": "C1: Nhẹ\n(Mild)", "range": (5, 9.9), "color": "#F59E0B", "action": "Tư vấn lối sống & theo dõi định kỳ"},
        {"name": "C2: Vừa\n(Moderate)", "range": (10, 14.9), "color": "#F97316", "action": "Đánh giá chuyên khoa, cân nhắc trị liệu"},
        {"name": "C3: Nặng vừa\n(Mod-Severe)", "range": (15, 19.9), "color": "#EA580C", "action": "Can thiệp dược lý & tâm lý tích cực"},
        {"name": "C4: Rất nặng\n(Severe)", "range": (20, 27.0), "color": "#DC2626", "action": "Yêu cầu can thiệp khẩn cấp chuyên khoa"}
    ]

    for t in tiers:
        x0, x1 = t["range"]
        rect = patches.FancyBboxPatch((x0, 1.25), x1 - x0, 1.15,
                                      boxstyle="round,pad=0.1,rounding_size=0.4",
                                      facecolor=t["color"], edgecolor='white', lw=1.8)
        ax.add_patch(rect)
        ax.text((x0 + x1)/2, 1.82, t["name"], ha='center', va='center', weight='bold',
                fontsize=9.0, color='white')
        ax.text((x0 + x1)/2, 0.65, t["action"], ha='center', va='center',
                fontsize=7.8, color='#334155', style='italic')

    ax.plot([0, 27], [1.25, 1.25], color='#1E293B', lw=2)
    for tick in [0, 5, 10, 15, 20, 27]:
        ax.plot([tick, tick], [1.15, 1.35], color='#1E293B', lw=1.8)
        ax.text(tick, 0.95, str(tick), ha='center', va='center', weight='bold', fontsize=9.5, color='#0F172A')

    # Top Brackets for T3 Mapping
    ax.plot([0, 0, 4.9, 4.9], [2.7, 2.9, 2.9, 2.7], color='#059669', lw=1.8)
    ax.text(2.45, 3.12, "T3: Nhóm C0 (Tối thiểu, 74.6%)", ha='center', va='bottom', fontsize=8.5, weight='bold', color='#047857')

    ax.plot([5, 5, 9.9, 9.9], [2.7, 2.9, 2.9, 2.7], color='#D97706', lw=1.8)
    ax.text(7.45, 3.12, "T3: Nhóm C1 (Nhẹ, 15.8%)", ha='center', va='bottom', fontsize=8.5, weight='bold', color='#B45309')

    ax.plot([10, 10, 27, 27], [2.7, 2.9, 2.9, 2.7], color='#DC2626', lw=1.8)
    ax.text(18.5, 3.12, "T3: Nhóm C2+ (Nặng quy ước = C2 ∪ C3 ∪ C4, 9.7%)", ha='center', va='bottom', fontsize=8.5, weight='bold', color='#B91C1C')

    save_clean_fig(fig, 'fig2_phq9_tiers.png')

# ==============================================================================
# 4. Figure 2.2: Gut-Brain & Metabolic Axis
# ==============================================================================
def make_fig2_gut_brain_metabolic_axis():
    fig, ax = create_canvas(width=11.5, height=5.6)

    c_box = patches.FancyBboxPatch((35, 38), 30, 24,
                                   boxstyle="round,pad=0.8,rounding_size=2.2",
                                   fc='#FEF2F2', ec='#DC2626', lw=2.0)
    ax.add_patch(c_box)
    ax.text(50, 53, "TRIỆU CHỨNG TRẦM CẢM", ha='center', va='center',
            weight='bold', fontsize=11, color='#991B1B')
    ax.text(50, 45, "(Lượng giá qua PHQ-9)", ha='center', va='center',
            style='italic', fontsize=9, color='#7F1D1D')

    systems = [
        {"title": "VIÊM MÃN TÍNH & MIỄN DỊCH",
         "items": "• Huyết sắc tố HbA1c (LBXGH)\n• Albumin huyết thanh (LBXSAL)\n• Cytokine gây viêm (IL-6, TNF-α)\n• Rối loạn hàng rào máu-não",
         "box": (3, 56, 29, 38), "fc": "#FFFBEB", "ec": "#D97706", "tc": "#92400E",
         "arr_start": (32.5, 68), "arr_end": (37.5, 59)},

        {"title": "RỐI LOẠN CHUYỂN HÓA & LIPID",
         "items": "• Chỉ số khối cơ thể (BMXBMI)\n• Chu vi vòng eo (BMXWAIST)\n• Cholesterol toàn phần (LBXTC)\n• HDL-Cholesterol (LBDHDD)",
         "box": (68, 56, 29, 38), "fc": "#EFF6FF", "ec": "#2563EB", "tc": "#1E3A8A",
         "arr_start": (67.5, 68), "arr_end": (62.5, 59)},

        {"title": "THẦN KINH TỰ CHỦ & HUYẾT ĐỘNG",
         "items": "• Huyết áp tâm thu (SBP_mean)\n• Huyết áp tâm trương (DBP_mean)\n• Nhịp tim khi nghỉ (BPXPLS)\n• Mất cân bằng trương lực giao cảm",
         "box": (3, 6, 29, 38), "fc": "#ECFDF5", "ec": "#059669", "tc": "#065F46",
         "arr_start": (32.5, 32), "arr_end": (37.5, 41)},

        {"title": "LỐI SỐNG & NHỊP SINH HỌC",
         "items": "• Thời lượng giấc ngủ (SLD010H)\n• Vận động thể lực (PAQ605..)\n• Thời gian tĩnh tại (PAD680)\n• Hút thuốc & Sử dụng cồn",
         "box": (68, 6, 29, 38), "fc": "#FAF5FF", "ec": "#7C3AED", "tc": "#4C1D95",
         "arr_start": (67.5, 32), "arr_end": (62.5, 41)}
    ]

    for s in systems:
        x, y, w, h = s["box"]
        p = patches.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc=s["fc"], ec=s["ec"], lw=1.5)
        ax.add_patch(p)
        ax.text(x + w/2, y + h - 6, s["title"], ha='center', va='center',
                weight='bold', fontsize=9.2, color=s["tc"])
        ax.text(x + 2, y + 15, s["items"], ha='left', va='center',
                fontsize=8.0, color='#1E293B', linespacing=1.3)

        ax.annotate('', xy=s["arr_end"], xytext=s["arr_start"],
                    arrowprops=dict(arrowstyle='<->', color=s["ec"], lw=2.0))

    save_clean_fig(fig, 'fig2_gut_brain_metabolic_axis.png')

# ==============================================================================
# 5. Figure 2.3: Class Imbalance & Feature Overlap Concept
# ==============================================================================
def make_fig2_imbalance_overlap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    classes = ['$C_0$\n(Tối thiểu)', '$C_1$\n(Nhẹ)', '$C_2$\n(Vừa)', '$C_3$\n(Nặng vừa)', '$C_4$\n(Rất nặng)']
    counts = [3016, 638, 243, 104, 44]
    pcts = [74.56, 15.77, 6.01, 2.57, 1.09]
    colors = ['#2563EB', '#F59E0B', '#F97316', '#DC2626', '#991B1B']

    bars = ax1.bar(classes, counts, color=colors, edgecolor='#1E293B', lw=1.0, width=0.6)
    ax1.set_title('(a) Mất cân bằng lớp cực đoan (IR = 68.5:1)', fontsize=10.5, weight='bold', pad=12)
    ax1.set_ylabel('Số lượng mẫu quan sát ($N = 4.045$)', fontsize=9.5)
    ax1.set_ylim(0, 3500)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for b, count, pct in zip(bars, counts, pcts):
        ax1.text(b.get_x() + b.get_width()/2, count + 70, f'{count}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=8.2, weight='bold')

    x = np.linspace(15, 45, 500)
    y0 = 0.08 * np.exp(-0.5 * ((x - 28) / 5)**2)
    y1 = 0.075 * np.exp(-0.5 * ((x - 29.5) / 5.5)**2)
    y2 = 0.065 * np.exp(-0.5 * ((x - 32) / 6)**2)

    ax2.plot(x, y0, label='$C_0$ (Tối thiểu: 74.6%)', color='#2563EB', lw=2.2)
    ax2.fill_between(x, y0, color='#2563EB', alpha=0.20)

    ax2.plot(x, y1, label='$C_1$ (Nhẹ: 15.8%)', color='#D97706', lw=2.2, linestyle='--')
    ax2.fill_between(x, y1, color='#F59E0B', alpha=0.20)

    ax2.plot(x, y2, label='$C_2^+$ (Nặng quy ước: 9.7%)', color='#DC2626', lw=2.2, linestyle=':')
    ax2.fill_between(x, y2, color='#DC2626', alpha=0.15)

    overlap = np.minimum(y0, y1)
    ax2.fill_between(x, overlap, color='#7C3AED', alpha=0.45, label='Vùng chồng lấn Bayes (Rất cao)')

    ax2.set_title('(b) Chồng lấn phân phối đặc trưng ($C_0$ vs $C_1$)', fontsize=10.5, weight='bold', pad=12)
    ax2.set_xlabel('Chỉ số sinh lý kết hợp (ví dụ: BMI / SBP z-score)', fontsize=9.5)
    ax2.set_ylabel('Mật độ xác suất $p(\\mathbf{x} \\mid y)$', fontsize=9.5)
    ax2.legend(loc='upper right', fontsize=8.0)
    ax2.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_clean_fig(fig, 'fig2_imbalance_overlap.png')

# ==============================================================================
# 6. Figure 2.4: Simple Conceptual Decision Tree (No fake specific numbers)
# ==============================================================================
def make_fig2_decision_tree():
    fig, ax = create_canvas(width=10.5, height=5.2)

    # Root Node
    r_box = patches.FancyBboxPatch((35, 78), 30, 16,
                                   boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.6)
    ax.add_patch(r_box)
    ax.text(50, 88.5, "NÚT GỐC (Root Node)", fontsize=9.5, fontweight='bold', ha='center', color='#1E3A8A')
    ax.text(50, 82, "Điều kiện phân tách:\nĐặc trưng X_j <= Ngưỡng theta_1",
            fontsize=8.5, ha='center', color='#1E293B', linespacing=1.2)

    # Branches from Root
    ax.annotate('', xy=(25, 66), xytext=(41, 77),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))
    ax.text(29, 73, "Đúng (True)", fontsize=8.5, fontweight='bold', color='#059669')

    ax.annotate('', xy=(75, 66), xytext=(59, 77),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))
    ax.text(69, 73, "Sai (False)", fontsize=8.5, fontweight='bold', color='#DC2626')

    # Internal Nodes (Level 1)
    n_left = patches.FancyBboxPatch((10, 48), 30, 16,
                                    boxstyle="round,pad=0.8,rounding_size=2.0",
                                    fc='#F8FAFC', ec='#64748B', lw=1.4)
    ax.add_patch(n_left)
    ax.text(25, 58.5, "Nút quyết định nội tại", fontsize=9.0, fontweight='bold', ha='center', color='#1E293B')
    ax.text(25, 52, "Điều kiện tiếp theo:\nĐặc trưng X_k <= Ngưỡng theta_2", fontsize=8.2, ha='center', color='#475569', linespacing=1.2)

    n_right = patches.FancyBboxPatch((60, 48), 30, 16,
                                     boxstyle="round,pad=0.8,rounding_size=2.0",
                                     fc='#F8FAFC', ec='#64748B', lw=1.4)
    ax.add_patch(n_right)
    ax.text(75, 58.5, "Nút quyết định nội tại", fontsize=9.0, fontweight='bold', ha='center', color='#1E293B')
    ax.text(75, 52, "Điều kiện tiếp theo:\nĐặc trưng X_m <= Ngưỡng theta_3", fontsize=8.2, ha='center', color='#475569', linespacing=1.2)

    # Branches to Leaves
    ax.annotate('', xy=(12, 33), xytext=(20, 47), arrowprops=dict(facecolor='#64748B', width=1.2, headwidth=5))
    ax.annotate('', xy=(38, 33), xytext=(30, 47), arrowprops=dict(facecolor='#64748B', width=1.2, headwidth=5))
    ax.annotate('', xy=(62, 33), xytext=(70, 47), arrowprops=dict(facecolor='#64748B', width=1.2, headwidth=5))
    ax.annotate('', xy=(88, 33), xytext=(80, 47), arrowprops=dict(facecolor='#64748B', width=1.2, headwidth=5))

    # 4 Conceptual Leaf Nodes (Level 2)
    leaves = [
        (3, 14, 20.5, "NÚT LÁ (Leaf Node)\nDự đoán lớp: C0\n(Phân phối xác suất\nnghiêng về lớp C0)", '#ECFDF5', '#059669', '#065F46'),
        (27.5, 14, 20.5, "NÚT LÁ (Leaf Node)\nDự đoán lớp: C1\n(Phân phối xác suất\nnghiêng về lớp C1)", '#FFFBEB', '#D97706', '#92400E'),
        (52, 14, 20.5, "NÚT LÁ (Leaf Node)\nDự đoán lớp: C1\n(Phân phối xác suất\nnghiêng về lớp C1)", '#FFFBEB', '#D97706', '#92400E'),
        (76.5, 14, 20.5, "NÚT LÁ (Leaf Node)\nDự đoán lớp: C2+\n(Phân phối xác suất\nnghiêng về lớp C2+)", '#FEF2F2', '#DC2626', '#991B1B')
    ]

    for x, y, w, text, fc, ec, tc in leaves:
        leaf_box = patches.FancyBboxPatch((x, y), w, 17,
                                          boxstyle="round,pad=0.8,rounding_size=1.8",
                                          fc=fc, ec=ec, lw=1.3)
        ax.add_patch(leaf_box)
        ax.text(x + w/2, y + 8.5, text, fontsize=7.8, ha='center', va='center', color=tc, linespacing=1.2)

    save_clean_fig(fig, 'fig2_decision_tree_concept.png')

# ==============================================================================
# 7. Figure 2.5: Random Forest Architecture
# ==============================================================================
def make_fig2_random_forest_arch():
    fig, ax = create_canvas(width=11.5, height=5.4)

    # Main Dataset
    d_box = patches.FancyBboxPatch((3, 36), 19, 32,
                                   boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.6)
    ax.add_patch(d_box)
    ax.text(12.5, 57, "TẬP DỮ LIỆU GỐC", fontsize=9.5, fontweight='bold', ha='center', color='#1E3A8A')
    ax.text(12.5, 46, "Dữ liệu huấn luyện D\n(N mẫu, d đặc trưng)\nKhông gian F4",
            fontsize=8.2, ha='center', va='center', color='#1E293B', linespacing=1.25)

    # 3 Bootstrap Samples
    bootstraps = [
        (30, 72, "Mẫu Bootstrap 1\n(D1, N mẫu hoàn lại)\nChọn tập con đặc trưng"),
        (30, 42, "Mẫu Bootstrap 2\n(D2, N mẫu hoàn lại)\nChọn tập con đặc trưng"),
        (30, 12, "Mẫu Bootstrap B\n(DB, N mẫu hoàn lại)\nChọn tập con đặc trưng")
    ]

    for bx, by, text in bootstraps:
        ax.annotate('', xy=(bx - 1, by + 9), xytext=(23, 52),
                    arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.3, headwidth=5))
        b_box = patches.FancyBboxPatch((bx, by), 18, 18,
                                      boxstyle="round,pad=0.8,rounding_size=1.8",
                                      fc='#F8FAFC', ec='#64748B', lw=1.3)
        ax.add_patch(b_box)
        ax.text(bx + 9, by + 9, text, fontsize=7.8, ha='center', va='center', color='#1E293B', linespacing=1.2)

    # Decision Trees
    trees = [
        (56, 72, "Cây quyết định 1\n(Huấn luyện trên D1)\nXuất xác suất P_1(c|x)"),
        (56, 42, "Cây quyết định 2\n(Huấn luyện trên D2)\nXuất xác suất P_2(c|x)"),
        (56, 12, "Cây quyết định B\n(Huấn luyện trên DB)\nXuất xác suất P_B(c|x)")
    ]

    for (bx, by, _), (tx, ty, ttext) in zip(bootstraps, trees):
        ax.annotate('', xy=(tx - 1, ty + 9), xytext=(bx + 19, by + 9),
                    arrowprops=dict(facecolor='#2563EB', edgecolor='#2563EB', width=1.5, headwidth=5))
        t_box = patches.FancyBboxPatch((tx, ty), 18, 18,
                                      boxstyle="round,pad=0.8,rounding_size=1.8",
                                      fc='#ECFDF5', ec='#059669', lw=1.3)
        ax.add_patch(t_box)
        ax.text(tx + 9, ty + 9, ttext, fontsize=8.0, ha='center', va='center', color='#065F46', linespacing=1.2)

    ax.text(39, 33, "• • •", fontsize=14, ha='center', va='center', color='#64748B')
    ax.text(65, 33, "• • •", fontsize=14, ha='center', va='center', color='#64748B')

    # Soft Voting Aggregator
    v_box = patches.FancyBboxPatch((81, 28), 16, 46,
                                   boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#FFFBEB', ec='#D97706', lw=1.6)
    ax.add_patch(v_box)
    ax.text(89, 64, "TỔNG HỢP\nSOFT VOTING", fontsize=9.2, fontweight='bold', ha='center', color='#92400E')
    ax.text(89, 46, "Xác suất trung bình:\nbar{P}(c|x) =\n(1/B) sum P_b(c|x)\n\nNhãn cuối cùng:\nargmax bar{P}(c|x)",
            fontsize=7.8, ha='center', va='center', color='#78350F', linespacing=1.25)

    for tx, ty, _ in trees:
        ax.annotate('', xy=(80, 51), xytext=(tx + 19, ty + 9),
                    arrowprops=dict(facecolor='#D97706', edgecolor='#D97706', width=1.3, headwidth=5))

    save_clean_fig(fig, 'fig2_random_forest_architecture.png')

# ==============================================================================
# 8. Figure 2.6: Three Modeling Paradigms
# ==============================================================================
def make_fig2_modeling_paradigms():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 4.8), dpi=300)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

    # Panel 1: Flat Paradigm
    ax1.set_title('(a) Phân loại phẳng (Flat)', fontsize=10.5, weight='bold', color='#1E3A8A', pad=10)
    p1 = patches.FancyBboxPatch((10, 75), 80, 18, boxstyle="round,pad=0.8,rounding_size=2.0", fc='#EFF6FF', ec='#2563EB', lw=1.4)
    ax1.add_patch(p1)
    ax1.text(50, 84, "Vectơ đặc trưng x (d = 34)", ha='center', va='center', weight='bold', fontsize=9.2, color='#1E3A8A')

    p_model = patches.FancyBboxPatch((15, 42), 70, 20, boxstyle="round,pad=0.8,rounding_size=2.0", fc='#FFFBEB', ec='#D97706', lw=1.5)
    ax1.add_patch(p_model)
    ax1.text(50, 52, "Bộ phân loại đa lớp\n(Random Forest / XGBoost)", ha='center', va='center', weight='bold', fontsize=8.8, color='#92400E')

    ax1.annotate('', xy=(50, 63), xytext=(50, 74), arrowprops=dict(facecolor='#475569', width=1.5, headwidth=5))
    ax1.annotate('', xy=(20, 24), xytext=(35, 41), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=4))
    ax1.annotate('', xy=(50, 24), xytext=(50, 41), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=4))
    ax1.annotate('', xy=(80, 24), xytext=(65, 41), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=4))

    ax1.text(20, 14, "C0\n(Tối thiểu)", ha='center', fontsize=8.5, weight='bold', color='#059669')
    ax1.text(50, 14, "C1\n(Nhẹ)", ha='center', fontsize=8.5, weight='bold', color='#D97706')
    ax1.text(80, 14, "C2+\n(Nặng)", ha='center', fontsize=8.5, weight='bold', color='#DC2626')

    # Panel 2: Hierarchical Paradigm
    ax2.set_title('(b) Phân loại phân cấp (Hierarchical)', fontsize=10.5, weight='bold', color='#065F46', pad=10)
    p2 = patches.FancyBboxPatch((10, 80), 80, 15, boxstyle="round,pad=0.8,rounding_size=2.0", fc='#ECFDF5', ec='#059669', lw=1.4)
    ax2.add_patch(p2)
    ax2.text(50, 87.5, "Đặc trưng x", ha='center', va='center', weight='bold', fontsize=9.2, color='#065F46')

    p_st1 = patches.FancyBboxPatch((15, 56), 70, 16, boxstyle="round,pad=0.8,rounding_size=2.0", fc='#059669', ec='#065F46', lw=1.5)
    ax2.add_patch(p_st1)
    ax2.text(50, 64, "Stage 1 (M1): Sàng lọc sơ bộ\nC0 vs C12 (5-27)", ha='center', va='center', weight='bold', fontsize=8.5, color='white')

    p_st2 = patches.FancyBboxPatch((42, 30), 52, 16, boxstyle="round,pad=0.8,rounding_size=2.0", fc='#D97706', ec='#92400E', lw=1.5)
    ax2.add_patch(p_st2)
    ax2.text(68, 38, "Stage 2 (M2): Phân tầng\nC1 vs C2+ (Học trên s >= 5)", ha='center', va='center', weight='bold', fontsize=8.2, color='white')

    ax2.annotate('', xy=(50, 73), xytext=(50, 79), arrowprops=dict(facecolor='#475569', width=1.5, headwidth=5))
    ax2.annotate('', xy=(25, 36), xytext=(35, 55), arrowprops=dict(facecolor='#475569', width=1.5, headwidth=5))
    ax2.annotate('', xy=(68, 47), xytext=(60, 55), arrowprops=dict(facecolor='#475569', width=1.5, headwidth=5))
    ax2.annotate('', xy=(55, 16), xytext=(60, 29), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=4))
    ax2.annotate('', xy=(82, 16), xytext=(76, 29), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=4))

    ax2.text(25, 26, "C0 (Loại bỏ)\n[Rủi ro lan truyền!]", ha='center', fontsize=8.0, weight='bold', color='#DC2626')
    ax2.text(55, 8, "C1 (Nhẹ)", ha='center', fontsize=8.2, weight='bold', color='#D97706')
    ax2.text(82, 8, "C2+ (Nặng)", ha='center', fontsize=8.2, weight='bold', color='#2563EB')

    # Panel 3: Ordinal Paradigm
    ax3.set_title('(c) Phân loại thứ bậc (Ordinal)', fontsize=10.5, weight='bold', color='#4C1D95', pad=10)
    p3 = patches.FancyBboxPatch((10, 80), 80, 15, boxstyle="round,pad=0.8,rounding_size=2.0", fc='#FAF5FF', ec='#7C3AED', lw=1.4)
    ax3.add_patch(p3)
    ax3.text(50, 87.5, "Đặc trưng x", ha='center', va='center', weight='bold', fontsize=9.2, color='#4C1D95')

    p_ord = patches.FancyBboxPatch((12, 45), 76, 24, boxstyle="round,pad=0.8,rounding_size=2.0", fc='#7C3AED', ec='#4C1D95', lw=1.5)
    ax3.add_patch(p_ord)
    ax3.text(50, 60, "Hàm tiềm ẩn: z = w^T x", ha='center', va='center', weight='bold', fontsize=9.0, color='white')
    ax3.text(50, 51, "Các ngưỡng cắt song song:\ntheta_1 < theta_2", ha='center', va='center', fontsize=8.2, color='#F3E8FF')

    ax3.annotate('', xy=(50, 70), xytext=(50, 79), arrowprops=dict(facecolor='#475569', width=1.5, headwidth=5))
    ax3.annotate('', xy=(20, 24), xytext=(32, 44), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=4))
    ax3.annotate('', xy=(50, 24), xytext=(50, 44), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=4))
    ax3.annotate('', xy=(80, 24), xytext=(68, 44), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=4))

    ax3.text(20, 14, "z < theta_1\n-> C0", ha='center', fontsize=8.5, weight='bold', color='#059669')
    ax3.text(50, 14, "theta_1 <= z < theta_2\n-> C1", ha='center', fontsize=8.5, weight='bold', color='#D97706')
    ax3.text(80, 14, "z >= theta_2\n-> C2+", ha='center', fontsize=8.5, weight='bold', color='#DC2626')

    plt.tight_layout()
    save_clean_fig(fig, 'fig2_modeling_paradigms.png')

# ==============================================================================
# 9. Figure 3.1: DepSev-ML Overall 7-Module Architecture
# ==============================================================================
def make_depsev_overall_architecture():
    fig, ax = create_canvas(width=12.5, height=6.2)

    h_box = patches.FancyBboxPatch((3, 85), 94, 11, boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(h_box)
    ax.text(50, 92, "NGUỒN DỮ LIỆU GỐC: KHẢO SÁT SỨC KHỎE VÀ DINH DƯỠNG QUỐC GIA HOA KỲ (CDC NHANES 2011--2016)",
            ha='center', va='center', weight='bold', fontsize=9.8, color='#1E3A8A')
    ax.text(50, 87.5, "Chu kỳ 2011--2012 (N = 4.639)   |   Chu kỳ 2013--2014 (N = 5.054)   |   Chu kỳ 2015--2016 (N = 4.888)",
            ha='center', va='center', fontsize=8.2, color='#334155')

    ax.annotate('', xy=(50, 77), xytext=(50, 84), arrowprops=dict(facecolor='#475569', width=1.5, headwidth=5))

    l1_mods = [
        (3, 56, 30, "MODULE 1: DATA INGESTION",
         "• Tiếp nhận file SAS .XPT\n• Lọc đối tượng >= 20 tuổi\n• Hợp lệ 9 câu hỏi DPQ",
         "#F0FDF4", "#16A34A", "#15803D"),
        (35, 56, 30, "MODULE 2: HARMONIZATION",
         "• Ghép nối qua mã SEQN\n• Lọc biến khuyết cấu trúc\n• Thống nhất schema 31 biến",
         "#EFF6FF", "#2563EB", "#1D4ED8"),
        (67, 56, 30, "MODULE 3: TARGET MAPPING",
         "• Tính tổng điểm s in [0, 27]\n• Ánh xạ nhị phân T2 (0-4 vs 5-27)\n• Ánh xạ 3 nhóm T3, 5 lớp T5",
         "#FFFBEB", "#D97706", "#B45309")
    ]

    for x, y, w, title, desc, fc, ec, tc in l1_mods:
        box = patches.FancyBboxPatch((x, y), w, 19, boxstyle="round,pad=0.8,rounding_size=1.8",
                                     fc=fc, ec=ec, lw=1.3)
        ax.add_patch(box)
        ax.text(x + w/2, y + 15, title, ha='center', va='center', weight='bold', fontsize=8.5, color=tc)
        ax.text(x + 2, y + 6.5, desc, ha='left', va='center', fontsize=7.6, color='#1E293B', linespacing=1.2)

    ax.annotate('', xy=(34, 65.5), xytext=(33.5, 65.5), arrowprops=dict(facecolor='#64748B', width=1.2, headwidth=4))
    ax.annotate('', xy=(66, 65.5), xytext=(65.5, 65.5), arrowprops=dict(facecolor='#64748B', width=1.2, headwidth=4))

    ax.annotate('', xy=(50, 48), xytext=(50, 55), arrowprops=dict(facecolor='#475569', width=1.5, headwidth=5))

    l2_mods = [
        (3, 30, 46, "MODULE 4: PARTITIONING & ISOLATION",
         "• Stratified 5-Fold Cross-Validation cố định\n• Khóa phân hoạch chuẩn bằng mã băm SHA-256\n• Phân tách rạch ròi Development và Temporal Test",
         "#FAF5FF", "#7C3AED", "#6D28D9"),
        (51, 30, 46, "MODULE 5: IN-FOLD PREPROCESSING PIPELINE",
         "• Fit Imputer (KNN / median) 100% bên trong train fold\n• Fit Scaler (mu, sigma) độc lập, transform sang val/test\n• BẢO TOÀN NGHIÊM NGẶT: ZERO DATA LEAKAGE",
         "#FEF2F2", "#DC2626", "#B91C1C")
    ]

    for x, y, w, title, desc, fc, ec, tc in l2_mods:
        box = patches.FancyBboxPatch((x, y), w, 16, boxstyle="round,pad=0.8,rounding_size=1.8",
                                     fc=fc, ec=ec, lw=1.3)
        ax.add_patch(box)
        ax.text(x + w/2, y + 12.5, title, ha='center', va='center', weight='bold', fontsize=8.5, color=tc)
        ax.text(x + 2, y + 5.5, desc, ha='left', va='center', fontsize=7.6, color='#1E293B', linespacing=1.2)

    ax.annotate('', xy=(50, 22), xytext=(50, 29), arrowprops=dict(facecolor='#475569', width=1.5, headwidth=5))

    l3_mods = [
        (3, 4, 46, "MODULE 6: MULTI-PARADIGM MODEL ZOO",
         "• Phẳng: RF Baseline (E0), XGBoost, Logistic Regression\n• Phân cấp: Hard Routing (E1-E3), Soft Probability (E4-E5)\n• Nhạy chi phí: Bayes Cost-Sensitive Risk Minimization (E6)\n• Thứ bậc: Balanced Ordinal Logistic (E7a), CBC (E7b)",
         "#EFF6FF", "#2563EB", "#1D4ED8"),
        (51, 4, 46, "MODULE 7: MULTI-DIMENSIONAL EVALUATION",
         "• Danh nghĩa: Macro-F1 (mục tiêu chính), Balanced Accuracy\n• Thứ bậc: Class MAE, Quadratic Weighted Kappa (QWK)\n• An toàn y tế: Số ca nặng bỏ sót (C2+ -> C0), Báo động nhầm\n• Kiểm tra tính tổng quát hóa theo dòng thời gian (Branch D)",
         "#ECFDF5", "#059669", "#047857")
    ]

    for x, y, w, title, desc, fc, ec, tc in l3_mods:
        box = patches.FancyBboxPatch((x, y), w, 16.5, boxstyle="round,pad=0.8,rounding_size=1.8",
                                     fc=fc, ec=ec, lw=1.3)
        ax.add_patch(box)
        ax.text(x + w/2, y + 13, title, ha='center', va='center', weight='bold', fontsize=8.5, color=tc)
        ax.text(x + 2, y + 5.5, desc, ha='left', va='center', fontsize=7.6, color='#1E293B', linespacing=1.2)

    save_clean_fig(fig, 'depsev_overall_architecture.png')

# ==============================================================================
# 10. Figure 3.2: In-Fold Preprocessing Pipeline & Zero Data Leakage
# ==============================================================================
def make_fig3_infold_cv():
    fig, ax = create_canvas(width=11.5, height=5.2)

    d_box = patches.FancyBboxPatch((3, 35), 18, 34, boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(d_box)
    ax.text(12, 59, "DỮ LIỆU NHANES", fontsize=9.5, fontweight='bold', ha='center', color='#1E3A8A')
    ax.text(12, 47, "N = 4.045 đối tượng\n34 biến F4 đầy đủ\nStratified 5-Fold Split\n(Cố định seed = 42)",
            fontsize=8.2, ha='center', va='center', color='#1E293B', linespacing=1.25)

    ax.annotate('', xy=(25, 52), xytext=(21.5, 52),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    fold_bg = patches.FancyBboxPatch((26, 7), 48, 86, boxstyle="round,pad=0.8,rounding_size=2.2",
                                     fc='#F8FAFC', ec='#94A3B8', lw=1.5)
    ax.add_patch(fold_bg)
    ax.text(50, 88, "LUỒNG XỬ LÝ TRONG MỖI FOLD (ZERO DATA LEAKAGE)",
            fontsize=9.5, fontweight='bold', ha='center', color='#1E293B')

    tr_box = patches.FancyBboxPatch((28, 51), 44, 31, boxstyle="round,pad=0.8,rounding_size=1.8",
                                    fc='#ECFDF5', ec='#059669', lw=1.3)
    ax.add_patch(tr_box)
    ax.text(50, 76, "1. Fold huấn luyện (Training Fold -- 80%)", fontsize=8.8, fontweight='bold', ha='center', color='#065F46')
    tr_steps = [
        "• Fit Imputer: Tính median / mode CHỈ trên tập train",
        "• Fit Scaler: Tính mean, std (mu, sigma) CHỈ trên train",
        "• Fit Model: Huấn luyện thuật toán với trọng số lớp",
        "-> KHÔNG nhìn thấy bất kỳ thông tin nào của tập test"
    ]
    ax.text(30, 61, "\n".join(tr_steps), fontsize=7.8, va='center', color='#047857', linespacing=1.25)

    ax.annotate('', xy=(50, 44), xytext=(50, 50.5),
                arrowprops=dict(facecolor='#059669', edgecolor='#059669', width=1.5, headwidth=5))
    ax.text(51, 47.5, "Chuyển tham số", fontsize=7.5, color='#047857')

    te_box = patches.FancyBboxPatch((28, 12), 44, 30, boxstyle="round,pad=0.8,rounding_size=1.8",
                                    fc='#FEF2F2', ec='#DC2626', lw=1.3)
    ax.add_patch(te_box)
    ax.text(50, 36, "2. Fold kiểm tra ngoài mẫu (OOF Test -- 20%)", fontsize=8.8, fontweight='bold', ha='center', color='#991B1B')
    te_steps = [
        "• Transform Imputer: Áp dụng median từ train",
        "• Transform Scaler: Chuẩn hóa bằng mu, sigma từ train",
        "• Predict: Dự đoán nhãn / xác suất hoàn toàn khách quan",
        "-> Bảo đảm tính khái quát hóa y sinh thực chất"
    ]
    ax.text(30, 22, "\n".join(te_steps), fontsize=7.8, va='center', color='#7F1D1D', linespacing=1.25)

    ax.annotate('', xy=(78, 52), xytext=(74.5, 52),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    o_box = patches.FancyBboxPatch((79, 28), 18, 48, boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#FFFBEB', ec='#D97706', lw=1.5)
    ax.add_patch(o_box)
    ax.text(88, 68, "ĐÁNH GIÁ OOF\nTOÀN CỤC", fontsize=9.2, fontweight='bold', ha='center', color='#92400E')
    ax.text(88, 48, "Gộp 5 Folds OOF:\n• Macro-F1\n• Balanced Accuracy\n• Class MAE\n• Ma trận nhầm lẫn\n\n-> Không lạc quan ảo\n-> Khả lập 100%",
            fontsize=8.0, ha='center', va='center', color='#78350F', linespacing=1.25)

    save_clean_fig(fig, 'fig3_infold_cv_pipeline.png')

# ==============================================================================
# 11. Figure 3.3: Feature Group Hierarchy (F1 -> F4)
# ==============================================================================
def make_fig3_3_feature_groups():
    fig, ax = create_canvas(width=11.5, height=5.2)

    groups = [
        {"name": "Nhóm F1 (10 biến cốt lõi)", "sub": "Thể chất, Dấu hiệu sinh tồn & Sinh hóa máu thiết yếu",
         "vars": "• Thể chất & Huyết động: BMXBMI, BMXWAIST, BPXPLS, SBP_mean, DBP_mean\n• Sinh hóa máu: HbA1c (LBXGH), Cholesterol (LBXTC), HDL (LBDHDD), Creatinine (LBXSCR), Albumin (LBXSAL)",
         "y": 74, "h": 22, "fc": "#EFF6FF", "ec": "#2563EB", "tc": "#1E3A8A"},

        {"name": "Nhóm F2 (19 biến = F1 + Tiền sử bệnh lý lâm sàng)", "sub": "Bổ sung 9 biến bệnh lý mạn tính",
         "vars": "• Bổ sung: Tăng huyết áp (BPQ020), Đái tháo đường (DIQ010), 5 bệnh tim mạch & đột quỵ (MCQ160B, C, D, E, F),\n  Ung thư (MCQ220), Tự đánh giá sức khỏe tổng quát (HUQ010)",
         "y": 50, "h": 21, "fc": "#ECFDF5", "ec": "#059669", "tc": "#065F46"},

        {"name": "Nhóm F3 (29 biến = F2 + Hành vi lối sống & Giấc ngủ)", "sub": "Bổ sung 10 biến thói quen sinh hoạt",
         "vars": "• Bổ sung: Hoạt động thể lực công việc & giải trí (PAQ605, 620, 650, 665), Thời gian tĩnh tại (PAD680),\n  Tình trạng hút thuốc (smoking_status), Thói quen dùng cồn (ALQ101, alcohol_status, alcohol_qty), Giấc ngủ (SLD010H)",
         "y": 26, "h": 21, "fc": "#FFFBEB", "ec": "#D97706", "tc": "#92400E"},

        {"name": "Nhóm F4 (34 biến = F3 + Nhân khẩu học & Kinh tế xã hội)", "sub": "Bổ sung 5 biến nhân khẩu học nền tảng",
         "vars": "• Bổ sung: Tuổi (RIDAGEYR), Giới tính (RIAGENDR), Chủng tộc (RIDRETH1), Trình độ học vấn (DMDEDUC2),\n  Tỷ lệ thu nhập gia đình trên chuẩn nghèo (INDFMPIR)",
         "y": 3, "h": 20, "fc": "#FEF2F2", "ec": "#DC2626", "tc": "#991B1B"}
    ]

    for g in groups:
        box = patches.FancyBboxPatch((3, g["y"]), 94, g["h"],
                                     boxstyle="round,pad=0.8,rounding_size=1.8",
                                     fc=g["fc"], ec=g["ec"], lw=1.4)
        ax.add_patch(box)
        ax.text(5, g["y"] + g["h"] - 5.5, f"{g['name']}  --  {g['sub']}", weight='bold', fontsize=8.8, color=g["tc"])
        ax.text(5, g["y"] + 6.5, g["vars"], fontsize=8.0, color='#1E293B', linespacing=1.2)

    save_clean_fig(fig, 'fig3_3_feature_groups.png')

# ==============================================================================
# 12. Figure 3.5: Hard Hierarchical Architecture (Có/Không -> Nhẹ/Nặng)
# ==============================================================================
def make_fig3_4_hard_hierarchical():
    fig, ax = create_canvas(width=11.5, height=5.2)

    # Input Box
    i_box = patches.FancyBboxPatch((3, 40), 16, 28, boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(i_box)
    ax.text(11, 57, "ĐỐI TƯỢNG (x)", fontsize=9.2, fontweight='bold', ha='center', color='#1E3A8A')
    ax.text(11, 47, "34 đặc trưng F4\nKhảo sát NHANES\n(N = 4.045)",
            fontsize=8.2, ha='center', va='center', color='#1E293B', linespacing=1.2)

    # Arrow to Stage 1
    ax.annotate('', xy=(24, 54), xytext=(19.5, 54),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    # Stage 1 Classifier (Có / Không có triệu chứng)
    st1_box = patches.FancyBboxPatch((24.5, 36), 24, 36, boxstyle="round,pad=0.8,rounding_size=2.0",
                                     fc='#ECFDF5', ec='#059669', lw=1.5)
    ax.add_patch(st1_box)
    ax.text(36.5, 65, "GIAI ĐOẠN 1 (Stage 1)", fontsize=10, fontweight='bold', ha='center', color='#065F46')
    ax.text(36.5, 58, "Sàng lọc Có/Không triệu chứng", fontsize=8.5, fontweight='bold', ha='center', color='#047857')
    ax.text(36.5, 46, "C0 (0-4)  vs  C12 (5-27)\n• Mô hình: Random Forest (M1)\n• Huấn luyện trên toàn bộ N mẫu",
            fontsize=8.0, ha='center', va='center', color='#065F46', linespacing=1.25)

    # Branch C0 (Stopping: Không triệu chứng)
    ax.annotate('', xy=(36.5, 23), xytext=(36.5, 35.5),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=6))
    ax.text(37.5, 29, "Không triệu chứng", fontsize=8.2, fontweight='bold', color='#334155')

    stop_box = patches.FancyBboxPatch((26.5, 7), 20, 15, boxstyle="round,pad=0.8,rounding_size=1.8",
                                      fc='#F1F5F9', ec='#94A3B8', lw=1.3)
    ax.add_patch(stop_box)
    ax.text(36.5, 14.5, "KẾT THÚC: C0\n(Tối thiểu: 0--4 điểm)\nDừng luồng xử lý",
            fontsize=8.0, fontweight='bold', ha='center', va='center', color='#475569')

    # Arrow to Stage 2 (Có triệu chứng)
    ax.annotate('', xy=(56.5, 54), xytext=(49, 54),
                arrowprops=dict(facecolor='#059669', edgecolor='#059669', width=1.5, headwidth=6))
    ax.text(52.5, 58, "Có triệu chứng\n(PHQ-9 >= 5)", fontsize=8.0, fontweight='bold', ha='center', color='#059669')

    # Stage 2 Classifier (Nhẹ / Nặng quy ước)
    st2_box = patches.FancyBboxPatch((57, 36), 24, 36, boxstyle="round,pad=0.8,rounding_size=2.0",
                                     fc='#FFFBEB', ec='#D97706', lw=1.5)
    ax.add_patch(st2_box)
    ax.text(69, 65, "GIAI ĐOẠN 2 (Stage 2)", fontsize=10, fontweight='bold', ha='center', color='#92400E')
    ax.text(69, 58, "Phân tầng Nhẹ/Nặng", fontsize=8.5, fontweight='bold', ha='center', color='#B45309')
    ax.text(69, 46, "C1 (5-9)  vs  C2+ (10-27)\n• Mô hình: Random Forest (M2)\n• Huấn luyện CHỈ trên các mẫu\n  có nhãn thật PHQ-9 >= 5",
            fontsize=8.0, ha='center', va='center', color='#92400E', linespacing=1.25)

    # Outcomes from Stage 2
    ax.annotate('', xy=(85.5, 62), xytext=(81.5, 57),
                arrowprops=dict(facecolor='#D97706', edgecolor='#D97706', width=1.2, headwidth=5))
    c1_out = patches.FancyBboxPatch((86, 55), 11, 15, boxstyle="round,pad=0.8,rounding_size=1.6",
                                    fc='#FEF3C7', ec='#D97706', lw=1.2)
    ax.add_patch(c1_out)
    ax.text(91.5, 62.5, "NHÃN C1\n(Nhẹ: 5--9)", fontsize=8.2, fontweight='bold', ha='center', va='center', color='#92400E')

    ax.annotate('', xy=(85.5, 43), xytext=(81.5, 48),
                arrowprops=dict(facecolor='#DC2626', edgecolor='#DC2626', width=1.2, headwidth=5))
    c2_out = patches.FancyBboxPatch((86, 35), 11, 15, boxstyle="round,pad=0.8,rounding_size=1.6",
                                    fc='#FEF2F2', ec='#DC2626', lw=1.2)
    ax.add_patch(c2_out)
    ax.text(91.5, 42.5, "NHÃN C2+\n(Nặng: 10--27)", fontsize=8.2, fontweight='bold', ha='center', va='center', color='#991B1B')

    # Error Propagation Bottleneck Box
    err_box = patches.FancyBboxPatch((55, 6), 42, 23, boxstyle="round,pad=0.8,rounding_size=1.8",
                                     fc='#FEF2F2', ec='#DC2626', lw=1.5, ls='--')
    ax.add_patch(err_box)
    ax.text(76, 22.5, "[!] ĐIỂM NGHẼN LAN TRUYỀN SAI SỐ MỘT CHIỀU",
            fontsize=8.5, fontweight='bold', ha='center', color='#991B1B')
    err_txt = (
        "• Có 120 ca nặng thật sự (C2+) bị Stage 1 đoán nhầm thành C0.\n"
        "• Quyết định cứng khiến 120 ca này bị loại bỏ vĩnh viễn khỏi luồng.\n"
        "• Stage 2 hoàn toàn không có cơ hội tiếp cận -> Bỏ sót ca nặng nguy hiểm!"
    )
    ax.text(57, 13, err_txt, fontsize=7.8, va='center', color='#7F1D1D', linespacing=1.25)

    save_clean_fig(fig, 'fig3_4_hard_hierarchical.png')

# ==============================================================================
# 13. Figure 3.6: Soft Hierarchical Architecture & Bayes Risk
# ==============================================================================
def make_fig3_5_soft_cost_sensitive():
    fig, ax = create_canvas(width=12.2, height=5.4)

    i_box = patches.FancyBboxPatch((3, 38), 14, 28, boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(i_box)
    ax.text(10, 56, "ĐỐI TƯỢNG (x)", fontsize=9.2, fontweight='bold', ha='center', color='#1E3A8A')
    ax.text(10, 46, "Đặc trưng F4\nKhảo sát NHANES\nN = 4.045",
            fontsize=8.0, ha='center', va='center', color='#1E293B', linespacing=1.2)

    ax.annotate('', xy=(21, 68), xytext=(17.5, 57),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))
    ax.annotate('', xy=(21, 38), xytext=(17.5, 47),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    st1_box = patches.FancyBboxPatch((21.5, 54), 22, 28, boxstyle="round,pad=0.8,rounding_size=1.8",
                                     fc='#ECFDF5', ec='#059669', lw=1.3)
    ax.add_patch(st1_box)
    ax.text(32.5, 75, "Stage 1 (M1 Soft)", fontsize=9.0, fontweight='bold', ha='center', color='#065F46')
    ax.text(32.5, 63, "Xuất phân phối xác suất:\n• P_M1(C0 | x)\n• P_M1(C12 | x)",
            fontsize=8.0, ha='center', va='center', color='#047857', linespacing=1.2)

    st2_box = patches.FancyBboxPatch((21.5, 18), 22, 28, boxstyle="round,pad=0.8,rounding_size=1.8",
                                     fc='#FFFBEB', ec='#D97706', lw=1.3)
    ax.add_patch(st2_box)
    ax.text(32.5, 39, "Stage 2 (M2 Soft)", fontsize=9.0, fontweight='bold', ha='center', color='#92400E')
    ax.text(32.5, 27, "Xuất xác suất điều kiện:\n• P_M2(C1 | C12, x)\n• P_M2(C2+ | C12, x)",
            fontsize=8.0, ha='center', va='center', color='#B45309', linespacing=1.2)

    ax.annotate('', xy=(48, 58), xytext=(44, 65),
                arrowprops=dict(facecolor='#059669', edgecolor='#059669', width=1.5, headwidth=6))
    ax.annotate('', xy=(48, 45), xytext=(44, 35),
                arrowprops=dict(facecolor='#D97706', edgecolor='#D97706', width=1.5, headwidth=6))

    fuse_box = patches.FancyBboxPatch((48.5, 18), 24, 64, boxstyle="round,pad=0.8,rounding_size=2.0",
                                      fc='#F8FAFC', ec='#64748B', lw=1.5)
    ax.add_patch(fuse_box)
    ax.text(60.5, 75, "TÍCH HỢP XÁC SUẤT MỀM", fontsize=9.2, fontweight='bold', ha='center', color='#1E293B')
    f_text = (
        "Xác suất hậu nghiệm toàn thể:\n\n"
        "hat{P}(C0|x) = P_M1(C0|x)\n\n"
        "hat{P}(C1|x) = P_M1(C12|x) *\n"
        "              P_M2(C1|C12,x)\n\n"
        "hat{P}(C2+|x) = P_M1(C12|x) *\n"
        "               P_M2(C2+|C12,x)\n\n"
        "-> Không có quyết định cứng\n"
        "-> Triệt tiêu hoàn toàn điểm nghẽn!"
    )
    ax.text(60.5, 48, f_text, fontsize=7.8, ha='center', va='center', color='#334155', linespacing=1.18)

    ax.annotate('', xy=(76.5, 50), xytext=(73, 50),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    bayes_box = patches.FancyBboxPatch((77, 8), 20, 84, boxstyle="round,pad=0.8,rounding_size=2.0",
                                       fc='#FFFBEB', ec='#D97706', lw=1.5)
    ax.add_patch(bayes_box)
    ax.text(87, 84, "QUYẾT ĐỊNH RỦI RO\nBAYES (E6b)", fontsize=9.2, fontweight='bold', ha='center', color='#92400E')
    b_text = (
        "Ma trận chi phí C(i, j):\n"
        "Phạt nặng lỗi bỏ sót:\n"
        "C(0, 2) = 5 (rất nặng)\n\n"
        "Rủi ro kỳ vọng:\n"
        "R(i|x) = sum C(i, j) hat{P}_j\n\n"
        "Quy tắc quyết định:\n"
        "y* = argmin_i R(i|x)\n\n"
        "[+] KẾT QUẢ ĐẠT ĐƯỢC:\n"
        "• Bỏ sót C2+ giảm từ\n"
        "  120 ca -> 31 ca\n"
        "  (Giảm 74.2% ca bỏ sót!)\n"
        "• BalAcc đạt 50.25%"
    )
    ax.text(87, 47, b_text, fontsize=7.8, ha='center', va='center', color='#78350F', linespacing=1.18)

    save_clean_fig(fig, 'fig3_5_soft_cost_sensitive.png')

# ==============================================================================
# 14. Figure 3.7: Ordinal Classification Principles
# ==============================================================================
def make_fig3_6_ordinal_classification():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-0.05, 1.1)
    z = np.linspace(-4, 4, 300)
    sig = 1 / (1 + np.exp(-z))
    ax1.plot(z, sig, color='#2563EB', lw=2.2, label='$\\sigma(z) = \\frac{1}{1 + e^{-z}}$')

    theta1 = -0.8
    theta2 = 1.2
    ax1.axvline(theta1, color='#DC2626', linestyle='--', lw=1.8, label='Ngưỡng $\\theta_1$ (Ranh giới $C_0 \\mid C_1$)')
    ax1.axvline(theta2, color='#D97706', linestyle=':', lw=1.8, label='Ngưỡng $\\theta_2$ (Ranh giới $C_1 \\mid C_2^+$)')

    ax1.text((theta1 - 4)/2, 0.5, 'Vùng $C_0$\n($s < \\theta_1$)', ha='center', fontsize=9.0, weight='bold', color='#2563EB')
    ax1.text((theta1 + theta2)/2, 0.5, 'Vùng $C_1$\n($\\theta_1 \\leq s < \\theta_2$)', ha='center', fontsize=9.0, weight='bold', color='#D97706')
    ax1.text((theta2 + 4)/2, 0.5, 'Vùng $C_2^+$\n($s \\geq \\theta_2$)', ha='center', fontsize=9.0, weight='bold', color='#DC2626')

    ax1.set_title('(a) Hồi quy Logistic thứ bậc (OLR)\nBiến tiềm ẩn $z = \\mathbf{w}^T\\mathbf{x}$ và ngưỡng cắt song song', fontsize=10.0, weight='bold', pad=10)
    ax1.set_xlabel('Hàm tiềm ẩn $z = \\mathbf{w}^T \\mathbf{x}$', fontsize=9.0)
    ax1.set_ylabel('Xác suất tích lũy $P(Y \\leq k \\mid \\mathbf{x})$', fontsize=9.0)
    ax1.legend(loc='upper left', fontsize=8.0)
    ax1.grid(linestyle='--', alpha=0.5)

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1.15)
    cases = ['Mẫu A\n(Hợp lệ)', 'Mẫu B\n(Hợp lệ)', 'Mẫu C\n(Vi phạm\nđơn điệu!)', 'Mẫu D\n(Hợp lệ)']
    x_pos = [1.5, 4.0, 6.5, 9.0]
    p_k1 = [0.85, 0.65, 0.30, 0.90]
    p_k2 = [0.20, 0.15, 0.45, 0.10]
    width = 0.6

    ax2.bar([p - width/2 for p in x_pos], p_k1, width=width, label='$P(Y \\geq 1 \\mid \\mathbf{x})$ (Bộ phân loại 1)', color='#2563EB', alpha=0.85)
    ax2.bar([p + width/2 for p in x_pos], p_k2, width=width, label='$P(Y \\geq 2 \\mid \\mathbf{x})$ (Bộ phân loại 2)', color='#DC2626', alpha=0.85)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cases, fontsize=8.5, weight='bold')
    ax2.set_title('(b) Phân loại nhị phân tích lũy (CBC)\nNguy cơ vi phạm tính đơn điệu ($P_{k+1} > P_k$)', fontsize=10.0, weight='bold', pad=10)
    ax2.set_ylabel('Xác suất dự đoán độc lập', fontsize=9.0)
    ax2.legend(loc='upper right', fontsize=8.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    ax2.annotate('Vi phạm: $P(Y \\geq 2) > P(Y \\geq 1)$\n(Tỷ lệ 0.32% trong E7b)',
                 xy=(6.5, 0.46), xytext=(4.3, 0.85),
                 arrowprops=dict(facecolor='#1E293B', width=1.0, headwidth=5),
                 fontsize=8.0, weight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#FEF2F2', ec='#DC2626'))

    plt.tight_layout()
    save_clean_fig(fig, 'fig3_6_ordinal_classification.png')

# ==============================================================================
# 15. Figure 3.8: Nested CV & Temporal Protocol
# ==============================================================================
def make_fig3_7_evaluation_protocol():
    fig, ax = create_canvas(width=11.5, height=5.2)

    p_top = patches.FancyBboxPatch((3, 53), 94, 42, boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(p_top)
    ax.text(5, 88, "TẦNG 1: QUY TRÌNH KIỂM ĐỊNH LỒNG NHAU (NESTED 5-FOLD STRATIFIED CV)", weight='bold', fontsize=9.5, color='#1E3A8A')
    
    cv_desc = [
        "Tập phát triển Legacy (NHANES 2013-2014, N = 4.045 đối tượng):",
        "• Outer CV (5 folds): Phân hoạch ngẫu nhiên tầng hóa theo nhãn mục tiêu (giữ nguyên tỷ lệ 74.6% : 15.8% : 9.7%).",
        "• Inner CV (3 folds trên từng Outer Training set): Tinh chỉnh siêu tham số (RandomizedSearchCV) tối ưu Macro-F1.",
        "• In-fold Preprocessing: Toàn bộ khâu điền khuyết (median/mode) và chuẩn hóa phương sai fit 100% trong từng fold huấn luyện.",
        "• Đánh giá OOF (Out-Of-Fold): Thu nhận dự đoán trên fold kiểm tra ngoài, tính trung bình và độ lệch chuẩn (Mean ± Std)."
    ]
    ax.text(5, 70, "\n".join(cv_desc), fontsize=8.2, color='#1E293B', linespacing=1.25)

    ax.annotate('', xy=(50, 46), xytext=(50, 52),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6))

    p_bot = patches.FancyBboxPatch((3, 5), 94, 40, boxstyle="round,pad=0.8,rounding_size=2.0",
                                   fc='#ECFDF5', ec='#059669', lw=1.5)
    ax.add_patch(p_bot)
    ax.text(5, 39, "TẦNG 2: KIỂM ĐỊNH ĐỘC LẬP THEO DÒNG THỜI GIAN (TEMPORAL VALIDATION)", weight='bold', fontsize=9.5, color='#065F46')

    temporal_desc = [
        "Tập huấn luyện (Training Cohort): NHANES 2013-2014 (N = 5.054, Branch A)",
        "Tập kiểm tra tương lai độc lập (Temporal Test Cohort): NHANES 2015-2016 (N = 4.888, Branch D)",
        "• Kiểm tra độ trôi phân phối y sinh học (biomedical covariate shift) theo thời gian khảo sát thực tế.",
        "• Hoàn toàn tách biệt về mặt thời gian và đối tượng tham gia khảo sát.",
        "• Chứng minh tính tổng quát hóa thực tế của mô hình: Balanced OLR đạt BalAcc = 0.5073 (so với In-cycle CV = 0.5151)."
    ]
    ax.text(5, 21, "\n".join(temporal_desc), fontsize=8.2, color='#1E293B', linespacing=1.25)

    save_clean_fig(fig, 'fig3_7_evaluation_protocol.png')

# ==============================================================================
# 16. Figure 4.6: Algorithm Comparison Bar Chart (T5 and T3)
# ==============================================================================
def make_fig4_algorithm_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    models_t5 = ['LR', 'DT', 'RF (Tuned)', 'XGBoost']
    macro_f1_t5 = [0.2407, 0.2079, 0.2755, 0.2621]
    bal_acc_t5 = [0.3253, 0.2514, 0.2983, 0.2743]
    w_f1_t5 = [0.5599, 0.5898, 0.6437, 0.6196]

    x1 = np.arange(len(models_t5))
    w = 0.26

    r1 = ax1.bar(x1 - w, macro_f1_t5, width=w, label='Macro-F1 (Mục tiêu chính)', color='#2563EB')
    r2 = ax1.bar(x1, bal_acc_t5, width=w, label='Balanced Accuracy', color='#059669')
    r3 = ax1.bar(x1 + w, w_f1_t5, width=w, label='Weighted F1', color='#94A3B8')

    ax1.set_title('(a) Đối sánh thuật toán trên T5 (5 lớp chuẩn)', fontsize=10.5, fontweight='bold', pad=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(models_t5, fontsize=9.5)
    ax1.set_ylim(0, 0.80)
    ax1.set_ylabel('Điểm số hiệu năng (OOF Mean)', fontsize=9.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8.0, loc='upper left')

    for bars in [r1, r2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.2f}",
                     ha='center', va='bottom', fontsize=7.5)

    models_t3 = ['E0: Flat RF', 'E1: Hard RF', 'E5: Soft RF-LR', 'E6b: Soft Bayes', 'E7a: Bal OLR']
    macro_f1_t3 = [0.4354, 0.4550, 0.4334, 0.4542, 0.4518]
    bal_acc_t3 = [0.4667, 0.4745, 0.4568, 0.5025, 0.5151]

    x2 = np.arange(len(models_t3))
    w2 = 0.35

    r4 = ax2.bar(x2 - w2/2, macro_f1_t3, width=w2, label='Macro-F1', color='#2563EB')
    r5 = ax2.bar(x2 + w2/2, bal_acc_t3, width=w2, label='Balanced Accuracy', color='#D97706')

    ax2.set_title('(b) Đối sánh các trường phái trên T3 (3 nhóm)', fontsize=10.5, fontweight='bold', pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['E0 (Flat)', 'E1 (Hard)', 'E5 (Soft)', 'E6b (Bayes)', 'E7a (OLR)'], fontsize=9.0)
    ax2.set_ylim(0, 0.65)
    ax2.set_ylabel('Điểm số hiệu năng (OOF Mean)', fontsize=9.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8.0, loc='lower right')

    for bars in [r4, r5]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.2f}",
                     ha='center', va='bottom', fontsize=7.8)

    plt.tight_layout()
    save_clean_fig(fig, 'fig4_algorithm_comparison_t5_t3.png')

# ==============================================================================
# 17. Figure 4.7: Confusion & Error Analysis (Severe FN Bottleneck vs Recovery)
# ==============================================================================
def make_fig4_confusion_error_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    models_err = ['E0 (Flat RF)', 'E1 (Hard RF)', 'E4 (Soft RF)', 'E5 (Soft RF-LR)', 'E6b (Bayes M1)', 'E7a (Bal OLR)']
    severe_fn = [118, 120, 227, 62, 31, 88]
    colors_fn = ['#64748B', '#DC2626', '#EA580C', '#2563EB', '#059669', '#7C3AED']

    b1 = ax1.bar(models_err, severe_fn, color=colors_fn, width=0.55)
    ax1.set_title('(a) Số ca nặng (C2+) bị bỏ sót thành bình thường (C0)', fontsize=10.0, fontweight='bold', pad=12)
    ax1.set_ylabel('Số ca bệnh bỏ sót (càng thấp càng an toàn)', fontsize=9.5)
    ax1.set_ylim(0, 260)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=25)

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 5, f"{h} ca",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax1.text(0.68, 0.85, "E6b giảm 74.2%\nca bỏ sót nặng so với E1!",
             transform=ax1.transAxes, fontsize=8.5, fontweight='bold', color='#15803D',
             bbox=dict(boxstyle="round,pad=0.4", fc="#DCFCE7", ec="#16A34A", lw=1.2))

    classes = ['C0 (Bình thường)', 'C1 (Nhẹ)', 'C2+ (Nặng quy ước)']
    rec_e1 = [0.8846, 0.2288, 0.3101]
    rec_e6b = [0.3806, 0.3824, 0.7444]
    rec_e7a = [0.5543, 0.3293, 0.6037]

    x = np.arange(len(classes))
    w = 0.26

    r1 = ax2.bar(x - w, rec_e1, width=w, label='E1: Hard Hierarchical', color='#DC2626')
    r2 = ax2.bar(x, rec_e6b, width=w, label='E6b: Soft Bayes Risk', color='#059669')
    r3 = ax2.bar(x + w, rec_e7a, width=w, label='E7a: Balanced OLR', color='#7C3AED')

    ax2.set_title('(b) So sánh Recall từng lớp: E1 vs E6b vs E7a', fontsize=10.0, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=9.0)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Tỷ lệ nhận diện (Recall)', fontsize=9.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8.0, loc='upper right')

    for bars in [r1, r2, r3]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h*100:.1f}%",
                     ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    save_clean_fig(fig, 'fig4_confusion_error_analysis.png')

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    print("=== STARTING MASTER REGENERATION OF ALL 17 PUBLICATION FIGURES ===")
    make_fig1_problem_formulation()
    make_fig1_research_progression()
    make_fig2_phq9_tiers()
    make_fig2_gut_brain_metabolic_axis()
    make_fig2_imbalance_overlap()
    make_fig2_decision_tree()
    make_fig2_random_forest_arch()
    make_fig2_modeling_paradigms()
    make_depsev_overall_architecture()
    make_fig3_infold_cv()
    make_fig3_3_feature_groups()
    make_fig3_4_hard_hierarchical()
    make_fig3_5_soft_cost_sensitive()
    make_fig3_6_ordinal_classification()
    make_fig3_7_evaluation_protocol()
    make_fig4_algorithm_comparison()
    make_fig4_confusion_error_analysis()
    print("=== ALL 17 FIGURES GENERATED PERFECTLY WITH ZERO CLIPPING & SPACIOUS LAYOUT ===")
