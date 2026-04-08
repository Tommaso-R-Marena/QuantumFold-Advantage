"""Generate a single-page visual results summary."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

out = "/home/user/workspace/QuantumFold-Advantage/results_real"
q = np.load(f"{out}/quantum_metrics.npz")
c = np.load(f"{out}/classical_metrics.npz")

QC = "#7B2FBE"
CC = "#2196F3"

fig = plt.figure(figsize=(16, 14))
fig.suptitle("QuantumFold-Advantage — Experiment Results", fontsize=18, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35, top=0.93, bottom=0.06)

# ---- Row 1: Metric comparisons ----
metrics_to_plot = [
    ("rmsd", "RMSD (Å) ↓", True),
    ("tm_score", "TM-score ↑", False),
    ("lddt", "lDDT ↑", False),
]
for i, (key, label, lower_better) in enumerate(metrics_to_plot):
    ax = fig.add_subplot(gs[0, i])
    qv, cv = q[key], c[key]
    bp = ax.boxplot([qv, cv], positions=[0, 1], widths=0.5, patch_artist=True,
                    boxprops=dict(linewidth=1.2),
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(QC)
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor(CC)
    bp["boxes"][1].set_alpha(0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Quantum", "Classical"], fontsize=10)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label.split("(")[0].strip(), fontsize=12, fontweight="bold")
    # Annotate means
    for j, (vals, color) in enumerate([(qv, QC), (cv, CC)]):
        ax.scatter(j, np.mean(vals), color=color, zorder=5, s=60, marker="D", edgecolors="black")

# ---- Row 2: Per-protein TM improvement + GDT-TS + GDT-HA ----
ax = fig.add_subplot(gs[1, :2])
n = len(q["tm_score"])
diff = q["tm_score"] - c["tm_score"]
order = np.argsort(diff)[::-1]
colors = [QC if d > 0 else CC for d in diff[order]]
ax.bar(range(n), diff[order], color=colors, alpha=0.8, edgecolor="white")
ax.set_xticks(range(n))
ax.set_xticklabels([f"P{i}" for i in order], fontsize=9)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Δ TM-score (Q − C)", fontsize=11)
ax.set_title("Per-Protein TM-score Improvement", fontsize=12, fontweight="bold")

ax3 = fig.add_subplot(gs[1, 2])
gdt_metrics = {"GDT-TS": ("gdt_ts", q["gdt_ts"].mean(), c["gdt_ts"].mean()),
               "GDT-HA": ("gdt_ha", q["gdt_ha"].mean(), c["gdt_ha"].mean())}
x_pos = np.arange(len(gdt_metrics))
width = 0.35
qvals = [v[1] for v in gdt_metrics.values()]
cvals = [v[2] for v in gdt_metrics.values()]
ax3.bar(x_pos - width/2, qvals, width, color=QC, alpha=0.7, label="Quantum")
ax3.bar(x_pos + width/2, cvals, width, color=CC, alpha=0.7, label="Classical")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(list(gdt_metrics.keys()), fontsize=10)
ax3.set_ylabel("Score", fontsize=11)
ax3.set_title("GDT Scores", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)

# ---- Row 3: Summary table + training info ----
ax_table = fig.add_subplot(gs[2, :])
ax_table.axis("off")

# Build summary table
header = ["Metric", "Quantum (mean±std)", "Classical (mean±std)", "Δ (Q−C)", "Cohen's d", "Direction"]
rows = []
for key in ["rmsd", "tm_score", "gdt_ts", "gdt_ha", "lddt"]:
    qv, cv = q[key], c[key]
    diff_val = np.mean(qv) - np.mean(cv)
    d_arr = qv - cv
    sd = np.std(d_arr, ddof=1)
    cd = diff_val / sd if sd > 1e-10 else 0.0
    lower_better = key == "rmsd"
    if lower_better:
        direction = "Q better ✓" if diff_val < 0 else "C better"
    else:
        direction = "Q better ✓" if diff_val > 0 else ("Tied" if abs(diff_val) < 1e-6 else "C better")
    rows.append([
        key.upper().replace("_", "-"),
        f"{np.mean(qv):.4f} ± {np.std(qv):.4f}",
        f"{np.mean(cv):.4f} ± {np.std(cv):.4f}",
        f"{diff_val:+.4f}",
        f"{cd:+.3f}",
        direction,
    ])

table = ax_table.table(
    cellText=rows,
    colLabels=header,
    loc="center",
    cellLoc="center",
    colWidths=[0.10, 0.20, 0.20, 0.12, 0.12, 0.14],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)
# Style header
for j in range(len(header)):
    table[0, j].set_facecolor("#333333")
    table[0, j].set_text_props(color="white", fontweight="bold")
# Highlight quantum-better rows
for i, row in enumerate(rows):
    if "Q better" in row[-1]:
        for j in range(len(header)):
            table[i+1, j].set_facecolor("#E8D5F5")

ax_table.set_title("Statistical Summary", fontsize=12, fontweight="bold", pad=15)

fig.savefig(f"{out}/experiment_summary.png", dpi=150, bbox_inches="tight")
print("Saved experiment_summary.png")
