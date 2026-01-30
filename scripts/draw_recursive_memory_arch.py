"""
Generate architecture diagram for WeaverStyleCompressor recursive memory system.

Creates a two-panel diagram:
  - Top panel: Main architecture flow (single-level)
  - Bottom panel: Options/Variants table + Design notes

Output: /home/jovyan/MemGenWorkspace/ltpo_sub/recursive_memory_arch.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
C_BG = "#FFFFFF"
C_CONTEXT = "#E0E0E0"       # Gray - context
C_LATENT = "#FFF3E0"        # Orange - query latents
C_ATTN = "#E3F2FD"          # Blue - attention
C_MLP = "#E8F5E9"           # Green - MLP
C_OUTPUT = "#F3E5F5"        # Purple - output
C_CYCLE_BG = "#FAFAFA"      # Very light gray for cycle loop
C_CYCLE_BORDER = "#607D8B"  # Blue-gray for dashed border
C_ARROW = "#37474F"         # Dark gray for arrows
C_TABLE_HEADER = "#455A64"  # Dark blue-gray for table header
C_TABLE_ALT = "#F5F5F5"     # Alternating row color
C_NOTE_BG = "#FFFDE7"       # Light yellow for notes
C_NORM = "#FFF8E1"          # Light amber for norm boxes
C_STEP = "#FAFAFA"          # Very light for step boxes

FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

# ---------------------------------------------------------------------------
# Figure: 20 wide x 36 tall to give ample room
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 22, 38
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG, dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")
ax.set_facecolor(C_BG)

CX = 11.0  # horizontal center


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def box(x, y, w, h, text, fc, ec="#424242", fs=11, fw="normal",
        tc="#212121", ls="-", lw=1.5, zorder=2):
    """Draw rounded box at (x, y) with width w, height h. Returns None."""
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fs, fontfamily=FONT,
                fontweight=fw, color=tc, zorder=zorder + 1, linespacing=1.45)


def arrow_down(x, y_from, y_to, color=C_ARROW, lw=2.0):
    """Vertical downward arrow."""
    a = FancyArrowPatch(
        (x, y_from), (x, y_to),
        arrowstyle="->", color=color, linewidth=lw,
        mutation_scale=15, zorder=3,
    )
    ax.add_patch(a)


def arrow_to(x1, y1, x2, y2, color=C_ARROW, lw=2.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", color=color, linewidth=lw,
        mutation_scale=15, zorder=3,
    )
    ax.add_patch(a)


# ===================================================================
# TITLE
# ===================================================================
y = FIG_H - 1.0
ax.text(CX, y, "WeaverStyleCompressor Architecture",
        ha="center", va="center", fontsize=24, fontfamily=FONT,
        fontweight="bold", color="#1A237E")
y -= 0.7
ax.text(CX, y, "Recursive Memory Compression   (memgen/model/recursive_memory.py)",
        ha="center", va="center", fontsize=13, fontfamily=FONT, color="#546E7A")

# ===================================================================
# INPUT BOXES
# ===================================================================
y -= 1.6
inp_h = 1.2

# Context (left of center)
ctx_w = 5.4
ctx_x = CX - 6.0
box(ctx_x, y, ctx_w, inp_h,
    "Context Embeddings\n(B, L, 4096)  [frozen]",
    C_CONTEXT, fs=12, fw="bold")

# Query Latents (right of center)
ql_w = 5.8
ql_x = CX + 0.6
box(ql_x, y, ql_w, inp_h,
    "Query Latents\nnn.Parameter(8, 4096)",
    C_LATENT, ec="#E65100", fs=12, fw="bold")

# Small italic label below query latents (shifted right to avoid arrow overlap)
ax.text(ql_x + ql_w / 2, y - 0.2,
        "prompt_query_latents / inference_query_latents",
        ha="center", va="top", fontsize=9, fontfamily=FONT,
        color="#BF360C", style="italic", zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

# Arrows from inputs to expand
expand_y = y - 1.8
expand_h = 0.75
expand_w = 10.5
expand_x = CX - expand_w / 2

arrow_to(ctx_x + ctx_w / 2, y, CX - 1.5, expand_y + expand_h)
arrow_to(ql_x + ql_w / 2, y, CX + 1.5, expand_y + expand_h)

# Expand box
box(expand_x, expand_y, expand_w, expand_h,
    "z = query_latents.unsqueeze(0).expand(B, -1, -1).clone()",
    "#F5F5F5", ec="#9E9E9E", fs=11)

# Arrow to cycle
arrow_down(CX, expand_y, expand_y - 0.6)

# ===================================================================
# CYCLE LOOP (dashed border)
# ===================================================================
cycle_top = expand_y - 0.7
cycle_bot = cycle_top - 13.0   # generous height
cycle_w = 14.5
cycle_x = CX - cycle_w / 2

cycle_patch = FancyBboxPatch(
    (cycle_x, cycle_bot), cycle_w, cycle_top - cycle_bot,
    boxstyle="round,pad=0.3",
    facecolor=C_CYCLE_BG, edgecolor=C_CYCLE_BORDER,
    linewidth=2.5, linestyle=(0, (8, 4)), zorder=1,
)
ax.add_patch(cycle_patch)

# Cycle label
ax.text(cycle_x + 0.6, cycle_top - 0.2,
        "Compression Cycle   (repeat x max_cycles,  default = 10)",
        ha="left", va="top", fontsize=14, fontfamily=FONT,
        fontweight="bold", color=C_CYCLE_BORDER, zorder=5)

# --- Step 1: Concatenation ---
s1_y = cycle_top - 1.6
s1_h = 0.85
s1_w = 11.0
box(CX - s1_w / 2, s1_y, s1_w, s1_h,
    "combined = cat([context, z], dim=1)     (B, L+8, 4096)",
    C_STEP, ec="#78909C", fs=11.5, zorder=3)

arrow_down(CX, s1_y, s1_y - 0.55)

# --- Step 2: Self-Attention ---
s2_y = s1_y - 2.7
s2_h = 2.1
s2_w = 12.5
box(CX - s2_w / 2, s2_y, s2_w, s2_h,
    "LowRank Causal Self-Attention\n\n"
    "Q, K, V, O :  LowRankLinear(4096, 4096, rank=64)\n"
    "8 heads  x  512 dim/head          [2.10M params]",
    C_ATTN, ec="#1565C0", fs=12, fw="bold", zorder=3)

arrow_down(CX, s2_y, s2_y - 0.55)

# --- Step 3: Post-norm residual (attn) ---
s3_y = s2_y - 1.5
s3_h = 0.85
s3_w = 11.5
box(CX - s3_w / 2, s3_y, s3_w, s3_h,
    "combined = RMSNorm( combined + self_attn(combined) )",
    C_NORM, ec="#F9A825", fs=11.5, zorder=3)

arrow_down(CX, s3_y, s3_y - 0.55)

# --- Step 4: Extract z ---
s4_y = s3_y - 1.5
s4_h = 0.85
s4_w = 11.0
box(CX - s4_w / 2, s4_y, s4_w, s4_h,
    "z = combined[:, -num_latents:]          (B, 8, 4096)",
    C_STEP, ec="#78909C", fs=11.5, zorder=3)

arrow_down(CX, s4_y, s4_y - 0.55)

# --- Step 5: SwiGLU MLP ---
s5_y = s4_y - 2.7
s5_h = 2.1
s5_w = 12.5
box(CX - s5_w / 2, s5_y, s5_w, s5_h,
    "LowRank SwiGLU MLP\n\n"
    "gate, up :  LowRankLinear(4096, 10936, rank=128)\n"
    "down :  LowRankLinear(10936, 4096, rank=128)     [5.77M params]",
    C_MLP, ec="#2E7D32", fs=12, fw="bold", zorder=3)

arrow_down(CX, s5_y, s5_y - 0.55)

# --- Step 6: Post-norm residual (MLP) ---
s6_y = s5_y - 1.5
s6_h = 0.85
s6_w = 10.5
box(CX - s6_w / 2, s6_y, s6_w, s6_h,
    "z = RMSNorm( z + mlp(z) )",
    C_NORM, ec="#F9A825", fs=11.5, zorder=3)

# --- Loopback arrow (right side, inside the cycle box) ---
loop_x = cycle_x + cycle_w - 1.2
ax.annotate(
    "", xy=(loop_x, s1_y + s1_h / 2), xytext=(loop_x, s6_y + s6_h / 2),
    arrowprops=dict(
        arrowstyle="->,head_width=0.4,head_length=0.3",
        color="#B71C1C", lw=2.2,
        connectionstyle="arc3,rad=-0.3",
    ),
    zorder=4,
)
ax.text(loop_x + 0.6, (s1_y + s6_y) / 2 + 0.6,
        "next\ncycle",
        ha="left", va="center", fontsize=11, fontfamily=FONT,
        color="#B71C1C", fontweight="bold", zorder=5)

# Arrow from cycle bottom to output
arrow_down(CX, cycle_bot, cycle_bot - 0.6)

# ===================================================================
# OUTPUT BOX
# ===================================================================
out_y = cycle_bot - 1.7
out_h = 1.0
out_w = 7.0
box(CX - out_w / 2, out_y, out_w, out_h,
    "Memory Output   (B, 8, 4096)",
    C_OUTPUT, ec="#6A1B9A", fs=13, fw="bold")

# ===================================================================
# SECTION DIVIDER
# ===================================================================
div_y = out_y - 1.2
ax.plot([1.5, 20.5], [div_y, div_y], color="#BDBDBD", linewidth=1, zorder=2)

# ===================================================================
# CONFIGURATION VARIANTS TABLE
# ===================================================================
table_title_y = div_y - 0.7
ax.text(CX, table_title_y, "Configuration Variants",
        ha="center", va="center", fontsize=18, fontfamily=FONT,
        fontweight="bold", color="#1A237E")

# Table layout
col_widths = [3.5, 4.5, 5.0, 2.0]
table_w = sum(col_widths)
table_x = CX - table_w / 2

headers = ["Option", "Config Key(s)", "Description", "Params"]
rows = [
    ["Single-Level\n(default)",
     "max_cycles: 10",
     "Fixed 10 compression\ncycles",
     "7.93M"],
    ["Two-Level",
     "two_level: true\nl_cycles: 6\nmax_h_cycles: 5",
     "H-cycle(5) x L-cycle(6)\nEarly stop per H-cycle",
     "7.93M"],
    ["Full-Rank MLP",
     "full_rank_mlp: true",
     "nn.Linear replaces\nLowRank SwiGLU",
     "~16.8M"],
    ["Bidirectional",
     "bidirectional: true",
     "Non-causal attention\n(like TRM)",
     "7.93M"],
    ["Skip Projection",
     "skip_projection: true",
     "No reasoner-weaver\nprojection layers",
     "7.93M"],
    ["With Projection",
     "skip_projection: false",
     "+reasoner_to_weaver\n+weaver_to_reasoner proj",
     "~41.5M"],
]

# Compute per-row heights
def row_height(row_data, base=0.55, per_line=0.38):
    n_lines = max(d.count("\n") + 1 for d in row_data)
    return base + per_line * n_lines

row_heights = [row_height(r) for r in rows]

# Header
hdr_h = 0.65
hdr_top = table_title_y - 0.7
for i, (hdr, cw) in enumerate(zip(headers, col_widths)):
    cx = table_x + sum(col_widths[:i])
    rect = plt.Rectangle((cx, hdr_top - hdr_h), cw, hdr_h,
                          facecolor=C_TABLE_HEADER, edgecolor="white",
                          linewidth=1, zorder=3)
    ax.add_patch(rect)
    ax.text(cx + cw / 2, hdr_top - hdr_h / 2, hdr,
            ha="center", va="center", fontsize=11.5, fontfamily=FONT,
            fontweight="bold", color="white", zorder=4)

# Rows
cursor_y = hdr_top - hdr_h
for r, (row_data, rh) in enumerate(zip(rows, row_heights)):
    ry = cursor_y - rh
    bg = C_TABLE_ALT if r % 2 == 0 else C_BG
    for i, (cell, cw) in enumerate(zip(row_data, col_widths)):
        cx = table_x + sum(col_widths[:i])
        rect = plt.Rectangle((cx, ry), cw, rh,
                              facecolor=bg, edgecolor="#BDBDBD",
                              linewidth=0.8, zorder=3)
        ax.add_patch(rect)
        fs = 10.5 if i == 0 else 10
        fw = "bold" if i == 0 else "normal"
        ff = MONO if i == 1 else FONT
        ax.text(cx + cw / 2, ry + rh / 2, cell,
                ha="center", va="center", fontsize=fs, fontfamily=ff,
                fontweight=fw, color="#212121", zorder=4, linespacing=1.35)
    cursor_y = ry

table_bottom = cursor_y

# ===================================================================
# BOTTOM INFO BOXES (side by side, below table)
# ===================================================================
info_top = table_bottom - 1.2
info_h = 3.6
info_gap = 1.0  # gap between left and right boxes

# Parameter Breakdown (left)
pl_w = 9.5
pl_x = CX - pl_w - info_gap / 2
box(pl_x, info_top - info_h, pl_w, info_h, "",
    "#EDE7F6", ec="#7E57C2", lw=1.5)

ax.text(pl_x + 0.5, info_top - 0.3,
        "Parameter Breakdown  (default config)",
        ha="left", va="top", fontsize=12.5, fontfamily=FONT,
        fontweight="bold", color="#4527A0")

param_lines = [
    "LowRank Causal Self-Attention:",
    "  4 x LowRankLinear(4096,4096,r=64) = 2.10M",
    "LowRank SwiGLU MLP:",
    "  gate(4096->10936, r=128) = 1.92M",
    "  up  (4096->10936, r=128) = 1.92M",
    "  down(10936->4096, r=128) = 1.93M",
    "Query Latents: 2 x (8 x 4096) = 0.07M",
    "Total                         = 7.93M",
]
ax.text(pl_x + 0.5, info_top - 0.85,
        "\n".join(param_lines),
        ha="left", va="top", fontsize=10, fontfamily=MONO,
        color="#311B92", linespacing=1.55)

# Design Notes (right)
nr_w = 9.5
nr_x = CX + info_gap / 2
box(nr_x, info_top - info_h, nr_w, info_h, "",
    C_NOTE_BG, ec="#F9A825", lw=1.5)

ax.text(nr_x + 0.5, info_top - 0.3,
        "Key Design Notes",
        ha="left", va="top", fontsize=12.5, fontfamily=FONT,
        fontweight="bold", color="#E65100")

notes_lines = [
    "- z placed at END of sequence",
    "  -> causal mask lets z attend to ALL context",
    "- Context re-concatenated every cycle (not cached)",
    "- Post-norm residual: RMSNorm(x + f(x))",
    "  prevents magnitude drift over 10+ cycles",
    "- LowRankLinear: xavier_uniform init",
    "  (NOT zero init -- see Bug #15)",
    "- down.weight: kaiming_uniform",
    "  up.weight: xavier_uniform",
]
ax.text(nr_x + 0.5, info_top - 0.85,
        "\n".join(notes_lines),
        ha="left", va="top", fontsize=10, fontfamily=FONT,
        color="#424242", linespacing=1.55)

# ===================================================================
# SAVE
# ===================================================================
output_path = "/home/jovyan/MemGenWorkspace/ltpo_sub/recursive_memory_arch.png"
fig.savefig(output_path, dpi=130, bbox_inches="tight",
            facecolor=C_BG, edgecolor="none", pad_inches=0.4)
plt.close(fig)
print(f"Saved to: {output_path}")
