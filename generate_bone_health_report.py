import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle
import textwrap

# -------------------------------
# Sample dummy data
# -------------------------------
patient = {"id": "P123456", "age": 68, "gender": "Female", "baseline": "65-70, Female"}
score = 7.1
features = {
    "GLCM Entropy": -2.1,
    "Cort. Thick.": -1.9,
    "BMI": 1.3,
    "Age": 1.1,
    "Trab. Int.": -0.8
}
risk_factors = [
    ("BMI", "28.2", "Medium"),
    ("Smoking", "Yes", "High"),
    ("Family History", "No", "Low")
]

# -------------------------------
# Style
# -------------------------------
COLORS = {
    "bg": "#F7F9FB",
    "card": "#FFFFFF",
    "border": "#E0E0E0",
    "green": "#4CAF50",
    "red": "#E57373",
    "yellow": "#FBC02D",
    "text": "#333333"
}

sns.set_style("white")

# -------------------------------
# Helpers
# -------------------------------
def add_card(ax):
    ax.set_facecolor(COLORS["card"])
    card = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02,rounding_size=10",
        transform=ax.transAxes, fc=COLORS["card"], ec=COLORS["border"], lw=1
    )
    ax.add_patch(card)

def score_color(s):
    if s >= 8: return COLORS["green"]
    elif s >= 5: return COLORS["yellow"]
    return COLORS["red"]

# -------------------------------
# Attractive Gauge (no card)
# -------------------------------
def draw_gauge(ax, score):
    ax.axis("equal")
    ax.axis("off")
    ax.set_facecolor("none")

    # Expand limits to prevent clipping
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.2)

    angles = np.linspace(180, 0, 200)
    ax.plot(np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles)),
            lw=18, color="#E1E4E8", solid_capstyle="round", clip_on=False)

    value_angle = 180 - (score/10)*180
    arc = np.linspace(180, value_angle, 200)
    ax.plot(np.cos(np.deg2rad(arc)), np.sin(np.deg2rad(arc)),
            lw=18, color=score_color(score), solid_capstyle="round", clip_on=False)

    needle_angle = np.deg2rad(value_angle)
    ax.plot([0, np.cos(needle_angle)*0.85],
            [0, np.sin(needle_angle)*0.85],
            lw=3.5, color=COLORS["text"], clip_on=False)

    ax.add_patch(Circle((0, 0), 0.03, color=COLORS["text"], clip_on=False))

    ax.text(0, -0.12, f"{score:.1f}/10",
            ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0, 0.25, "Risk Level", ha="center", fontsize=10, color=COLORS["text"])

# -------------------------------
# Feature Chart
# -------------------------------
def draw_feature_chart(ax, features):
    add_card(ax)
    items = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [COLORS["green"] if v > 0 else COLORS["red"] for v in values]

    ax.barh(labels, values, color=colors, edgecolor="black", alpha=0.8)
    ax.axvline(0, color="#999", lw=1)
    ax.axvline(1, color="#ccc", lw=0.8, ls="--")
    ax.axvline(-1, color="#ccc", lw=0.8, ls="--")
    ax.axvline(2, color="#ddd", lw=0.8, ls=":")
    ax.axvline(-2, color="#ddd", lw=0.8, ls=":")
    ax.set_title("Feature Deviation (Z-scores)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Deviation from Baseline", fontsize=10)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(axis='x', alpha=0.2, color="#EEEEEE")

    ax.scatter([], [], color=COLORS["green"], label="Above baseline")
    ax.scatter([], [], color=COLORS["red"], label="Below baseline")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

# -------------------------------
# Risk Table
# -------------------------------
def draw_risk_table(ax, risk_factors):
    add_card(ax)
    ax.axis("off")
    ax.text(0.05, 0.88, "Risk Factor Analysis", fontsize=12, fontweight="bold")
    y = 0.65
    for factor, value, level in risk_factors:
        color = COLORS["red"] if level == "High" else COLORS["yellow"] if level == "Medium" else COLORS["green"]
        ax.text(0.05, y, factor, fontsize=10)
        ax.text(0.45, y, value, fontsize=10)
        ax.text(0.7, y, level, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=0.3))
        y -= 0.22

# -------------------------------
# Main Report
# -------------------------------
def build_report():
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = GridSpec(3, 2, height_ratios=[1.2, 1.5, 1.2], hspace=0.5, wspace=0.4)

    fig.suptitle("Bone Health Analysis Report", fontsize=18, fontweight="bold", y=0.98)

    # Patient Info card
    ax_info = fig.add_subplot(gs[0, 0])
    add_card(ax_info); ax_info.axis("off")
    ax_info.text(0.05, 0.8, f"Patient ID: {patient['id']}", fontsize=11)
    ax_info.text(0.05, 0.6, f"Age: {patient['age']}", fontsize=11)
    ax_info.text(0.05, 0.4, f"Gender: {patient['gender']}", fontsize=11)
    ax_info.text(0.05, 0.2, f"Baseline Group: {patient['baseline']}", fontsize=11)

    # Gauge (no card)
    ax_gauge = fig.add_subplot(gs[0, 1])
    draw_gauge(ax_gauge, score)

    # Feature chart card
    ax_feat = fig.add_subplot(gs[1, :])
    draw_feature_chart(ax_feat, features)

    # Interpretation card
    ax_interp = fig.add_subplot(gs[2, 0])
    add_card(ax_interp); ax_interp.axis("off")
    interp = ("Score indicates moderate risk. Most notable deviation is "
              "GLCM Entropy (-2.1 Z), which is a critical deviation.")
    ax_interp.text(0.05, 0.85, "Interpretation", fontsize=12, fontweight="bold")
    ax_interp.text(0.05, 0.55, "\n".join(textwrap.wrap(interp, 52)), fontsize=10)

    # Risk table card
    ax_risk = fig.add_subplot(gs[2, 1])
    draw_risk_table(ax_risk, risk_factors)

    fig.text(0.5, 0.02, "This report is AI-assisted and for clinical support only.",
             ha="center", fontsize=9, color="#888")

    plt.tight_layout()
    plt.savefig("bone_health_report.png", dpi=220, bbox_inches="tight")
    plt.close()
    print("Saved: bone_health_report.png")

if __name__ == "__main__":
    build_report()