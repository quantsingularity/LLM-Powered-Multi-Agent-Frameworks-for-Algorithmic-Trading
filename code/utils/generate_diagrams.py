"""
Generate architecture and sequence diagrams using matplotlib when graphviz is not available.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Architecture diagram
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(
    7,
    9.5,
    "LLM-Powered Multi-Agent Trading System Architecture",
    ha="center",
    fontsize=16,
    fontweight="bold",
)

# Data layer
data_box = FancyBboxPatch(
    (0.5, 7.5),
    3,
    1,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor="lightgray",
    linewidth=2,
)
ax.add_patch(data_box)
ax.text(
    2, 8, "Market Data\n(OHLCV, News, Macro)", ha="center", va="center", fontsize=10
)

# Agent layer
agents = [
    ("Analyst\nAgent", 1.5, 5.5, "lightblue"),
    ("Decision\nAgent", 4.5, 5.5, "lightblue"),
    ("Risk\nAgent", 7.5, 5.5, "orange"),
    ("Execution\nAgent", 10.5, 5.5, "lightgreen"),
]

for name, x, y, color in agents:
    box = FancyBboxPatch(
        (x - 0.75, y),
        1.5,
        1,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor=color,
        linewidth=2,
    )
    ax.add_patch(box)
    ax.text(x, y + 0.5, name, ha="center", va="center", fontsize=9, fontweight="bold")

# Explainability agent
box = FancyBboxPatch(
    (6, 3.5),
    2,
    1,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="yellow",
    linewidth=2,
)
ax.add_patch(box)
ax.text(
    7,
    4,
    "Explainability\nAgent",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
)

# RL Policy
box = FancyBboxPatch(
    (10, 3.5),
    2,
    1,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="pink",
    linewidth=2,
)
ax.add_patch(box)
ax.text(
    11,
    4,
    "RL Policy\n(PPO/DQN)",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
)

# Orchestrator
circle = mpatches.Circle(
    (7, 1.5), 0.6, edgecolor="black", facecolor="lightyellow", linewidth=2
)
ax.add_patch(circle)
ax.text(7, 1.5, "Orchestrator", ha="center", va="center", fontsize=9, fontweight="bold")

# Market
box = FancyBboxPatch(
    (10, 1),
    2,
    1,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="lightgray",
    linewidth=2,
)
ax.add_patch(box)
ax.text(11, 1.5, "Market\n(Orders)", ha="center", va="center", fontsize=10)

# Arrows
arrows = [
    (2, 7.5, 2, 6.6, "Features"),
    (2.25, 5.5, 3.75, 5.5, "Analysis"),
    (5.25, 5.5, 6.75, 5.5, "Decision"),
    (8.25, 5.5, 9.75, 5.5, "Approved"),
    (10.5, 4.6, 7.75, 4.6, "Executed"),
    (7, 3.5, 7, 2.2, "Explanation"),
    (11, 3.5, 7.5, 1.9, "Signals"),
    (11, 4.6, 5.5, 6.2, "RL Action"),
    (11, 1, 11, 1.4, "Orders"),
]

for x1, y1, x2, y2, label in arrows:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2,
        color="darkblue",
    )
    ax.add_patch(arrow)
    # Add label
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(
        mid_x + 0.3, mid_y + 0.1, label, fontsize=8, style="italic", color="darkred"
    )

plt.tight_layout()
plt.savefig(
    "/home/user/llm-trading-research/figures/architecture_diagram.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Created architecture diagram")

# Sequence diagram
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(
    7,
    9.5,
    "Agent Interaction Sequence (Single Trading Cycle)",
    ha="center",
    fontsize=16,
    fontweight="bold",
)

# Agents as vertical lines
agents_seq = [
    "Market\nData",
    "Analyst",
    "Decision",
    "Risk",
    "Execution",
    "Explainability",
    "RL Policy",
]
positions = [1, 3, 5, 7, 9, 11, 13]

for agent, pos in zip(agents_seq, positions):
    # Box at top
    box = FancyBboxPatch(
        (pos - 0.6, 8.5),
        1.2,
        0.8,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor="lightblue",
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(pos, 8.9, agent, ha="center", va="center", fontsize=9, fontweight="bold")

    # Vertical line (lifeline)
    ax.plot([pos, pos], [8.5, 0.5], "k--", linewidth=1, alpha=0.5)

# Message arrows (time flows downward)
messages = [
    (1, 3, 7.5, "Request Features"),
    (3, 5, 6.8, "Market Analysis"),
    (13, 5, 6.5, "RL Signal"),
    (5, 7, 6.0, "Trading Decision"),
    (7, 9, 5.3, "Approved Trade"),
    (9, 11, 4.6, "Execution Report"),
    (11, 13, 3.9, "Update Policy"),
    (9, 1, 3.2, "Submit Order"),
    (1, 9, 2.5, "Execution Confirm"),
    (11, 3, 1.8, "Performance Feedback"),
]

for x1, x2, y, label in messages:
    # Arrow
    arrow = FancyArrowPatch(
        (x1, y),
        (x2, y),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
        color="darkblue",
    )
    ax.add_patch(arrow)
    # Label
    mid_x = (x1 + x2) / 2
    ax.text(
        mid_x,
        y + 0.15,
        label,
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

# Time indicator
ax.text(0.2, 9, "Time ↓", fontsize=10, fontweight="bold", rotation=90, va="center")

plt.tight_layout()
plt.savefig(
    "/home/user/llm-trading-research/figures/sequence_diagram.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Created sequence diagram")

# Explainability example visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

ax.text(
    6,
    7.5,
    "Explainability Agent Output Example",
    ha="center",
    fontsize=16,
    fontweight="bold",
)

# Decision box
box = FancyBboxPatch(
    (0.5, 6),
    11,
    0.8,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="lightgreen",
    linewidth=2,
)
ax.add_patch(box)
ax.text(
    6,
    6.4,
    "Decision: BUY 0.25 (increase position by 25% of portfolio)",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
)

# Evidence section
ax.text(1, 5.5, "Supporting Evidence:", fontsize=12, fontweight="bold")

evidence_items = [
    ("Technical:", "RSI=28 (oversold), MACD crossover (bullish), Volume spike"),
    ("Sentiment:", "Positive news (0.6/1.0), 5 articles today, earnings beat"),
    ("Macro:", "Fed rates stable, VIX declining (15.2), yield curve normalizing"),
]

y_pos = 5.0
for category, details in evidence_items:
    ax.text(1.5, y_pos, f"{category}", fontsize=10, fontweight="bold")
    ax.text(3.5, y_pos, details, fontsize=9)
    y_pos -= 0.5

# Reasoning chain
ax.text(1, 3.5, "Reasoning Chain:", fontsize=12, fontweight="bold")

reasoning_flow = [
    "1. Technical oversold condition → Mean reversion opportunity",
    "2. Positive sentiment → Catalyst for price movement",
    "3. Favorable macro → Low systemic risk",
    "4. All signals aligned → High confidence trade",
]

y_pos = 3.0
for step in reasoning_flow:
    arrow_x = 1.3
    ax.arrow(
        arrow_x,
        y_pos,
        0.3,
        0,
        head_width=0.08,
        head_length=0.1,
        fc="darkblue",
        ec="darkblue",
    )
    ax.text(arrow_x + 0.5, y_pos, step, fontsize=9, va="center")
    y_pos -= 0.4

# Risk assessment
box = FancyBboxPatch(
    (0.5, 0.8),
    11,
    0.6,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="lightyellow",
    linewidth=2,
)
ax.add_patch(box)
ax.text(
    6,
    1.1,
    "Risks: Limited downside (stop-loss at -3%), Volatility may spike on news",
    ha="center",
    va="center",
    fontsize=10,
)

# Confidence
box = FancyBboxPatch(
    (0.5, 0.2),
    5,
    0.4,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="lightblue",
    linewidth=2,
)
ax.add_patch(box)
ax.text(
    3,
    0.4,
    "Confidence Level: HIGH (85%)",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig(
    "/home/user/llm-trading-research/figures/explainability_example.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Created explainability example")

print("\nAll diagrams generated successfully!")
