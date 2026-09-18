#!/usr/bin/env python3
"""Render every README figure from the JSON files in results/.

Each figure is written twice, as <name>-light.png and <name>-dark.png, so the
README can serve the right one for the reader's GitHub theme. Nothing here is
hard-coded: change a number in results/ and the figures follow.

Example:
    python scripts/make_figures.py                       # all figures -> assets/
    python scripts/make_figures.py --font-dir ~/fonts    # use Inter if you have it
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPI = 200

# Colour roles. Series hues keep a fixed order (blue, orange, aqua) and a fixed
# meaning across figures: blue = English / Latin / this model, orange = Odia,
# aqua = Dogri / Devanagari. Text always uses the ink colours, never a series hue.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "context": "#c3c2b7",
        "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "context": "#52514e",
        "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70",
    },
}


def load(name: str) -> dict:
    with open(os.path.join(REPO_ROOT, "results", name), encoding="utf-8") as f:
        return json.load(f)


def setup_fonts(font_dir):
    family = "DejaVu Sans"
    if font_dir:
        for path in glob.glob(os.path.join(os.path.expanduser(font_dir), "Inter*.[ot]tf")):
            font_manager.fontManager.addfont(path)
            family = "Inter"
    plt.rcParams.update({"font.family": family, "font.size": 10, "svg.fonttype": "path"})


def new_figure(theme: dict, width: float, height: float):
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    fig.patch.set_facecolor(theme["surface"])
    return fig


def style_axes(ax, theme: dict, grid_axis: str = "y"):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    baseline = "bottom" if grid_axis == "y" else "left"
    ax.spines[baseline].set_visible(True)
    ax.spines[baseline].set_color(theme["axis"])
    ax.spines[baseline].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=theme["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["muted"], labelsize=8.5, length=0, pad=5)


def title(fig, theme: dict, text: str, subtitle: str = None, x: float = 0.06, y: float = 0.93):
    fig.text(x, y, text, color=theme["ink"], fontsize=12.5, fontweight="semibold", va="top")
    if subtitle:
        fig.text(x, y - 0.42 / fig.get_figheight(), subtitle, color=theme["ink2"], fontsize=9.5, va="top")


def pixels_to_data(ax, pixels: float):
    """Size of `pixels` screen pixels (at 100 px per inch) in data units, as (dx, dy)."""
    ax.figure.canvas.draw()
    inverse = ax.transData.inverted()
    (x0, y0), (x1, y1) = inverse.transform([(0, 0), (pixels * DPI / 100, pixels * DPI / 100)])
    return abs(x1 - x0), abs(y1 - y0)


def hbar(ax, y: float, x0: float, x1: float, height: float, color: str, rounded: bool = True):
    """Horizontal bar from x0 to x1, square at the baseline end and rounded at the data end."""
    if x1 <= x0:
        return
    bottom, top = y - height / 2, y + height / 2
    if not rounded:
        ax.add_patch(Rectangle((x0, bottom), x1 - x0, height, color=color, linewidth=0))
        return
    rx, ry = pixels_to_data(ax, 4)
    rx, ry = min(rx, x1 - x0), min(ry, height / 2)
    vertices = [(x0, bottom), (x1 - rx, bottom), (x1, bottom), (x1, bottom + ry),
                (x1, top - ry), (x1, top), (x1 - rx, top), (x0, top), (x0, bottom)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(vertices, codes), color=color, linewidth=0))


def save(fig, out_dir: str, name: str, mode: str):
    path = os.path.join(out_dir, f"{name}-{mode}.png")
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- figures


def figure_hero(theme: dict):
    """KPI row: the five numbers that define the project."""
    params = load("model_params.json")
    corpus = load("corpus_stats.json")
    tokenizer = load("tokenizer_stats.json")
    pretrain = load("pretrain_log.json")
    hardware = load("hardware.json")

    tiles = [
        (f"{params['total_parameters'] / 1e6:.0f}M", "parameters", "trained from scratch"),
        (f"{corpus['total_tokens'] / 1e9:.2f}B", "training tokens", "English, Odia and Dogri"),
        (f"{tokenizer['compression']['odia']['token_ratio_baseline_over_ours']:.1f}×", "fewer tokens on Odia",
         "vs the stock Qwen3 tokenizer"),
        (f"{hardware['pretrain_gpu_hours']:.0f}", "H100 GPU-hours", "of pre-training, on 2 GPUs"),
        (f"{pretrain['eval'][-1]['eval_perplexity']:.1f}", "validation perplexity",
         f"on {pretrain['eval'][-1]['eval_samples'] * 4096 / 1e6:.0f}M held-out tokens"),
    ]

    fig = new_figure(theme, 10, 1.9)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    step = 0.94 / len(tiles)
    for i, (value, label, note) in enumerate(tiles):
        x = 0.045 + i * step
        if i:
            ax.plot([x - 0.022, x - 0.022], [0.22, 0.78], color=theme["grid"], linewidth=0.8)
        ax.text(x, 0.62, value, color=theme["ink"], fontsize=25, fontweight="semibold", va="center")
        ax.text(x, 0.34, label, color=theme["ink"], fontsize=9.5, fontweight="medium", va="center")
        ax.text(x, 0.20, note, color=theme["muted"], fontsize=7.8, va="center")
    return fig


def figure_pretraining(theme: dict):
    """Loss against tokens seen: the whole run, and a zoom on everything after the first 0.8B tokens."""
    log = load("pretrain_log.json")
    billions = log["tokens_per_step"] / 1e9
    train_x = [point["step"] * billions for point in log["train"]]
    train_y = [point["loss"] for point in log["train"]]
    eval_x = [point["step"] * billions for point in log["eval"]]
    eval_y = [point["eval_loss"] for point in log["eval"]]
    epoch_boundary = log["max_steps"] / log["num_train_epochs"] * billions

    fig = new_figure(theme, 10, 4.3)
    title(fig, theme, "Pre-training loss",
          "Cross-entropy per token against tokens processed. Validation is a held-out 1% of the packed corpus.")
    panels = [
        (fig.add_axes([0.06, 0.13, 0.40, 0.58]), "Whole run", (0, 4.9), (2.8, 10.2), 0),
        (fig.add_axes([0.55, 0.13, 0.40, 0.58]), "After the first 0.8B tokens", (0.7, 4.9), (3.1, 3.95), 0.78),
    ]
    for ax, name, xlim, ylim, start in panels:
        style_axes(ax, theme)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        points = [(x, y) for x, y in zip(train_x, train_y) if x >= start]
        ax.plot(*zip(*points), color=theme["blue"], linewidth=2, solid_capstyle="round", solid_joinstyle="round")
        ax.plot(eval_x, eval_y, color=theme["orange"], linewidth=2, marker="o", markersize=7.5,
                markeredgecolor=theme["surface"], markeredgewidth=1.6, solid_capstyle="round")
        ax.axvline(epoch_boundary, color=theme["axis"], linewidth=0.8)
        ax.text(epoch_boundary + 0.05, ylim[1] - 0.02 * (ylim[1] - ylim[0]), "epoch 2",
                color=theme["muted"], fontsize=8, va="top")
        ax.set_title(name, color=theme["ink2"], fontsize=9.5, loc="left", pad=8)
        ax.set_xlabel("tokens processed (billions)", color=theme["muted"], fontsize=8.5, labelpad=6)

    zoom = panels[1][0]
    zoom.annotate(f"{eval_y[-1]:.2f}  (perplexity {log['eval'][-1]['eval_perplexity']:.1f})",
                  xy=(eval_x[-1], eval_y[-1]), xytext=(eval_x[-1] - 0.1, eval_y[-1] + 0.2),
                  ha="right", color=theme["ink"], fontsize=9, fontweight="medium",
                  arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8,
                              "shrinkB": 5, "connectionstyle": "arc3,rad=-0.15"})
    handles = [Line2D([], [], color=theme["blue"], linewidth=2, label="training loss"),
               Line2D([], [], color=theme["orange"], linewidth=2, marker="o", markersize=6,
                      markeredgecolor=theme["surface"], label="validation loss")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.955, 0.965), ncol=2, frameon=False,
               fontsize=9, labelcolor=theme["ink2"], handlelength=1.8, columnspacing=1.6)
    return fig


def figure_corpus_and_vocabulary(theme: dict):
    """Two part-to-whole bars: what the model read, and how the vocabulary is spent."""
    corpus = load("corpus_stats.json")["by_language"]
    vocab = load("tokenizer_stats.json")["vocabulary_by_script"]
    other_vocab = sum(count for script, count in vocab.items() if script not in ("latin", "odia", "devanagari"))

    rows = [
        ("Training tokens", f"{sum(entry['tokens'] for entry in corpus.values()) / 1e9:.2f}B",
         [corpus["english"]["tokens"], corpus["odia"]["tokens"], corpus["dogri"]["tokens"], 0]),
        ("Vocabulary pieces", f"{sum(vocab.values()):,}",
         [vocab["latin"], vocab["odia"], vocab["devanagari"], other_vocab]),
    ]
    colors = [theme["blue"], theme["orange"], theme["aqua"], theme["context"]]
    labels = ["English · Latin script", "Odia", "Dogri · Devanagari", "digits and symbols"]

    fig = new_figure(theme, 10, 3.0)
    title(fig, theme, "English dominates the corpus, but not the vocabulary",
          "The tokenizer was trained on a similar amount of text per language, so over half of its pieces are Odia or Devanagari.")
    ax = fig.add_axes([0.20, 0.10, 0.75, 0.46])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 1.6)
    ax.axis("off")
    gap, _ = pixels_to_data(ax, 2)
    min_label_width, _ = pixels_to_data(ax, 44)

    for row, (name, total, values) in enumerate(rows):
        y = 1 - row
        fig.text(0.06, ax.get_position().y0 + ax.get_position().height * (y + 0.6) / 2.2, name,
                 color=theme["ink"], fontsize=10, fontweight="medium", va="bottom")
        fig.text(0.06, ax.get_position().y0 + ax.get_position().height * (y + 0.6) / 2.2 - 0.005, total,
                 color=theme["muted"], fontsize=8.5, va="top")
        shares = [100 * value / sum(values) for value in values]
        shown = [i for i, share in enumerate(shares) if share > 0]
        left = 0
        for i in shown:
            right = left + shares[i]
            last = i == shown[-1]
            hbar(ax, y, left, right if last else right - gap, 0.42, colors[i], rounded=last)
            text = f"{shares[i]:.1f}%"
            if shares[i] >= min_label_width:
                on_dark_fill = colors[i] in (theme["blue"], theme["orange"]) or theme is THEMES["dark"]
                ax.text((left + right) / 2, y, text, ha="center", va="center", fontsize=9, fontweight="medium",
                        color="#ffffff" if on_dark_fill else "#0b0b0b")
            else:
                ax.annotate(f"{labels[i].split(' ·')[0]} {text}", xy=((left + right) / 2, y + 0.21),
                            xytext=(right - 1.5, y + 0.52), ha="right", va="center", fontsize=8.5,
                            color=theme["ink2"],
                            arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8})
            left = right

    handles = [Line2D([], [], marker="s", markersize=8, linestyle="", color=color, label=label)
               for color, label in zip(colors, labels)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.19, 0.60), ncol=4, frameon=False,
               fontsize=9, labelcolor=theme["ink2"], handletextpad=0.3, columnspacing=1.6)
    return fig


def figure_tokenizer_efficiency(theme: dict):
    """Tokens per word: this tokenizer against the stock Qwen3 tokenizer."""
    stats = load("tokenizer_stats.json")
    languages = [("english", "English"), ("odia", "Odia"), ("dogri", "Dogri")]

    fig = new_figure(theme, 10, 3.8)
    title(fig, theme, "Tokens needed per word",
          f"The 32k tokenizer trained for this model against the {stats['baseline']['vocab_size'] // 1000}k-piece "
          "tokenizer that ships with Qwen3, on the same text.")
    ax = fig.add_axes([0.12, 0.14, 0.82, 0.50])
    style_axes(ax, theme, grid_axis="x")
    longest = max(stats["compression"][key]["baseline"]["tokens_per_word"] for key, _ in languages)
    ax.set_xlim(0, longest * 1.12)
    ax.set_ylim(-0.6, len(languages) - 0.4)
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels([name for _, name in reversed(languages)], color=theme["ink"], fontsize=10)
    ax.set_xlabel("tokens per whitespace-separated word (lower is better)", color=theme["muted"],
                  fontsize=8.5, labelpad=6)

    for position, (key, _) in enumerate(reversed(languages)):
        entry = stats["compression"][key]
        ours, baseline = entry["cognita_mini"]["tokens_per_word"], entry["baseline"]["tokens_per_word"]
        hbar(ax, position + 0.17, 0, ours, 0.26, theme["blue"])
        hbar(ax, position - 0.17, 0, baseline, 0.26, theme["context"])
        ax.text(ours + longest * 0.012, position + 0.17, f"{ours:.2f}", va="center", color=theme["ink"],
                fontsize=9, fontweight="medium")
        ratio = entry["token_ratio_baseline_over_ours"]
        note = f"  ({ratio:.1f}× as many)" if ratio > 1.05 else ""
        ax.text(baseline + longest * 0.012, position - 0.17, f"{baseline:.2f}{note}", va="center",
                color=theme["ink2"], fontsize=9)

    handles = [Line2D([], [], marker="s", markersize=8, linestyle="", color=theme["blue"],
                      label="Cognita-Mini tokenizer (32k)"),
               Line2D([], [], marker="s", markersize=8, linestyle="", color=theme["context"],
                      label="stock Qwen3 tokenizer")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.775), ncol=2, frameon=False,
               fontsize=9, labelcolor=theme["ink2"], handletextpad=0.3, columnspacing=1.6)
    return fig


def figure_finetuning(theme: dict):
    """Before and after instruction tuning, per task, with the trivial baseline marked."""
    tuned = load("char_count_eval.json")["by_dataset"]
    base = load("char_count_eval_base.json")["by_dataset"]
    trivial = load("char_count_baselines.json")
    piqa_tuned, piqa_base = load("piqa_eval.json"), load("piqa_eval_base.json")
    scripts = [("english", "English"), ("odia", "Odia script"), ("dogri_script", "Devanagari script")]

    fig = new_figure(theme, 10, 4.2)
    title(fig, theme, "What instruction tuning adds",
          "Pre-trained model against the same model with the LoRA adapter. "
          "The tick marks the best score reachable without reading the input.")
    panels = [
        (fig.add_axes([0.17, 0.14, 0.22, 0.42]), "Character counting · exact", "exact", "majority_answer"),
        (fig.add_axes([0.45, 0.14, 0.22, 0.42]), "Character counting · within ±2", "within_2", "best_constant_within_2"),
    ]
    for ax, name, metric, trivial_key in panels:
        style_axes(ax, theme, grid_axis="x")
        ax.set_xlim(-4, 100)
        ax.set_ylim(-0.6, len(scripts) - 0.4)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_yticks(range(len(scripts)))
        ax.set_yticklabels([label for _, label in reversed(scripts)] if ax is panels[0][0] else [],
                           color=theme["ink"], fontsize=9.5)
        ax.set_title(name, color=theme["ink2"], fontsize=9.5, loc="left", pad=8)
        for position, (key, _) in enumerate(reversed(scripts)):
            before, after = 100 * base[key][metric], 100 * tuned[key][metric]
            chance = 100 * trivial[key][trivial_key]
            ax.plot([before, after], [position, position], color=theme["context"], linewidth=2,
                    solid_capstyle="round", zorder=2)
            ax.plot([chance, chance], [position - 0.2, position + 0.2], color=theme["muted"], linewidth=1.4, zorder=3)
            ax.plot(before, position, "o", color=theme["context"], markersize=8.5,
                    markeredgecolor=theme["surface"], markeredgewidth=1.6, zorder=4)
            ax.plot(after, position, "o", color=theme["blue"], markersize=8.5,
                    markeredgecolor=theme["surface"], markeredgewidth=1.6, zorder=5)
            ax.text(after + 5, position, f"{after:.1f}%", va="center", color=theme["ink"], fontsize=9,
                    fontweight="medium")

    ax = fig.add_axes([0.76, 0.14, 0.19, 0.42])
    style_axes(ax, theme, grid_axis="x")
    ax.set_xlim(40, 70)
    ax.set_ylim(-0.6, len(scripts) - 0.4)
    ax.set_xticks([40, 50, 60, 70])
    ax.set_xticklabels(["40%", "50%", "60%", "70%"])
    ax.set_yticks([1])
    ax.set_yticklabels(["English"], color=theme["ink"], fontsize=9.5)
    ax.set_title("PIQA · two-way choice", color=theme["ink2"], fontsize=9.5, loc="left", pad=8)
    before = 100 * piqa_base["accuracy_length_normalised"]
    after = 100 * piqa_tuned["accuracy_length_normalised"]
    ax.plot([before, after], [1, 1], color=theme["context"], linewidth=2, solid_capstyle="round", zorder=2)
    ax.plot([50, 50], [0.8, 1.2], color=theme["muted"], linewidth=1.4, zorder=3)
    ax.plot(before, 1, "o", color=theme["context"], markersize=8.5, markeredgecolor=theme["surface"],
            markeredgewidth=1.6, zorder=4)
    ax.plot(after, 1, "o", color=theme["blue"], markersize=8.5, markeredgecolor=theme["surface"],
            markeredgewidth=1.6, zorder=5)
    ax.text(after + 1.6, 1, f"{after:.1f}%", va="center", color=theme["ink"], fontsize=9, fontweight="medium")
    ax.text(before, 0.62, f"{before:.1f}%", ha="center", va="center", color=theme["ink2"], fontsize=8.5)

    handles = [Line2D([], [], marker="o", markersize=7, linestyle="", color=theme["context"], label="pre-trained only"),
               Line2D([], [], marker="o", markersize=7, linestyle="", color=theme["blue"], label="with LoRA adapter"),
               Line2D([], [], marker="|", markersize=9, markeredgewidth=1.4, linestyle="", color=theme["muted"],
                      label="constant guess / chance")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.79), ncol=3, frameon=False,
               fontsize=9, labelcolor=theme["ink2"], handletextpad=0.3, columnspacing=1.6)
    return fig


FIGURES = {
    "hero": figure_hero,
    "pretraining": figure_pretraining,
    "corpus_and_vocabulary": figure_corpus_and_vocabulary,
    "tokenizer_efficiency": figure_tokenizer_efficiency,
    "finetuning": figure_finetuning,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "assets"))
    parser.add_argument("--font-dir", default=None, help="Folder containing Inter .otf/.ttf files (optional)")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="Render a subset")
    args = parser.parse_args()

    setup_fonts(args.font_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.only or FIGURES:
        for mode, theme in THEMES.items():
            print(save(FIGURES[name](theme), args.out_dir, name, mode))


if __name__ == "__main__":
    main()
