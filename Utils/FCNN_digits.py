import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

## set the color themes of the figures
style = open(os.path.join(os.path.dirname(__file__), "plot_style.py")).read()
exec(style)


def draw_fcnn_digit_classifier() -> None:
    """Plot a fully connected network classifying MNIST digits: 784 -> 32 -> 10 (softmax)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # X positions for layers
    x_input, x_hidden, x_output = 0.0, 1.0, 2.0

    # Y positions for nodes: input layer is shown as a truncated stack with a "..." gap
    n_input_shown = 18
    y_input = [i / (n_input_shown - 1) for i in range(n_input_shown)]
    gap_idx = n_input_shown // 2  # index used to draw the ellipsis, node omitted here

    n_hidden = 32
    y_hidden = [i / (n_hidden - 1) for i in range(n_hidden)]

    labels_output = [str(d) for d in range(10)]
    y_output = [i / (len(labels_output) - 1) for i in range(len(labels_output))]

    # Draw all fully connected edges (input -> hidden), skipping the omitted node
    for i, yi in enumerate(y_input):
        if i == gap_idx:
            continue
        for yh in y_hidden:
            ax.plot([x_input, x_hidden], [yi, yh], color="#00FFFF", alpha=0.08, linewidth=0.6)

    # Draw all fully connected edges (hidden -> output)
    for yh in y_hidden:
        for yo in y_output:
            ax.plot([x_hidden, x_output], [yh, yo], color="#FFD700", alpha=0.15, linewidth=0.6)

    # Draw nodes
    node_style = dict(s=70, edgecolors="white", linewidths=0.6, zorder=3)
    input_y_no_gap = [yi for i, yi in enumerate(y_input) if i != gap_idx]
    ax.scatter([x_input] * len(input_y_no_gap), input_y_no_gap, color="#00FF00", **node_style)
    y_gap = y_input[gap_idx]
    dy = (y_input[1] - y_input[0]) * 0.28
    ax.scatter([x_input] * 3, [y_gap - dy, y_gap, y_gap + dy], color="white", s=6, zorder=4)

    ax.scatter([x_hidden] * len(y_hidden), y_hidden, color="#FF00FF", s=45, edgecolors="white", linewidths=0.4, zorder=3)

    ax.scatter([x_output] * len(y_output), y_output, color="#FF3333", s=120, edgecolors="white", linewidths=0.8, zorder=3)
    halo = [pe.withStroke(linewidth=3, foreground="black")]
    for yo, label in zip(y_output, labels_output):
        ax.text(x_output + 0.12, yo, label, color="white", fontsize=12, fontweight="bold",
                 ha="left", va="center", zorder=4, path_effects=halo)

    ax.text(x_input, -0.08, "input\n(784 pixels)", color="white", fontsize=11,
             ha="center", va="top", path_effects=halo)
    ax.text(x_hidden, -0.08, "hidden\n(32 neurons)", color="white", fontsize=11,
             ha="center", va="top", path_effects=halo)
    ax.text(x_output, -0.08, "output\n(10 digits, softmax)", color="white", fontsize=11,
             ha="center", va="top", path_effects=halo)

    ax.set_xlim(-0.35, 2.55)
    ax.set_ylim(-0.22, 1.08)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "FCNN_digits.png"), bbox_inches="tight", dpi=150, transparent=True)
    plt.show()


if __name__ == "__main__":
    draw_fcnn_digit_classifier()
