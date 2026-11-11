"""Generate the assignment PDF from the recorded experiment evidence."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


REPOSITORY = "https://github.com/SAMSUNYUJUN/mlds-rl-assignment2"
INK = "#182C41"
MUTED = "#51667B"
BLUE = "#166895"
ORANGE = "#AD5324"
LIGHT = "#E2EAF1"
PALE = "#F2F7FA"
WIDTH = 510


def read_csv(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=float))


def read_json(path):
    return json.loads(path.read_text())


def evidence(path):
    training = read_csv(path / "training.csv")
    evaluation = read_csv(path / "evaluation.csv")
    metadata = read_json(path / "metadata.json")
    summary = read_json(path / "evaluation.json")
    validation = read_json(path / "validation.json")
    if len(evaluation) != 500 or summary["episodes"] != 500:
        raise ValueError(f"{path} needs exactly 500 evaluation episodes")
    if len(training) != metadata["episodes"] or len(training) != metadata["config"]["episodes"]:
        raise ValueError(f"{path} needs a complete training run")
    if summary["checkpoint_episode"] != validation["selected_episode"]:
        raise ValueError(f"{path} has evaluation and validation from different checkpoints")
    np.testing.assert_allclose([summary["mean"], summary["sd"]],
                               [evaluation["reward"].mean(), evaluation["reward"].std()])
    return training, evaluation, metadata, summary, validation


def chart_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.2, "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": LIGHT,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": LIGHT, "grid.linewidth": 0.65, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8.3, "savefig.dpi": 210,
    })


def episode_charts(path, data):
    training, evaluation, meta, summary, _ = data
    episodes = training["episode"]
    raw = training["reward"]
    peak = training["max_q"]
    selected = summary["checkpoint_episode"]
    moving = np.convolve(raw, np.ones(100) / 100, mode="valid")
    scores = evaluation["reward"]
    fig, axes = plt.subplots(3, 1, figsize=(7.13, 6.13), layout="constrained")
    ax_q, ax_r, ax_h = axes
    ax_q.plot(episodes, peak, color=BLUE, lw=0.85, alpha=0.8)
    ax_q.set(ylabel="Peak Q", xlabel="Training episode")
    ax_q.set_title("Maximum Q-value during each episode", loc="left", pad=8)
    ax_r.plot(episodes, raw, color="#7AACC5", lw=0.65, alpha=0.55, label="Episode reward")
    ax_r.plot(episodes[99:], moving, color=BLUE, lw=1.8, label="Last 100 episodes")
    ax_r.set(ylabel="Raw reward", xlabel="Training episode")
    ax_r.set_title("Training rewards", loc="left", pad=8)
    for ax in (ax_q, ax_r):
        ax.axvline(selected, color=ORANGE, lw=1.2, ls="--", label=f"Checkpoint {selected:,}")
        ax.set_xlim(1, meta["episodes"])
        ax.xaxis.set_major_locator(MaxNLocator(8, integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax_r.legend(loc="upper left", ncols=3, columnspacing=1.2)
    if meta["config"]["game"] == "cartpole" and np.ptp(scores) <= 30:
        bins = np.arange(min(scores.min() - 1, 496) - 0.5, 501.5, 1)
    else:
        bins = 24
    ax_h.hist(scores, bins=bins, color=BLUE, alpha=0.82, edgecolor="white", linewidth=0.5)
    ax_h.axvline(summary["mean"], color=ORANGE, lw=1.3, ls="--", label=f"Mean {summary['mean']:,.2f}")
    ax_h.set(ylabel="Episodes", xlabel="Raw reward")
    ax_h.set_title("Greedy evaluation rewards across 500 episodes", loc="left", pad=8)
    ax_h.legend(loc="upper left")
    ax_h.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for ax in axes:
        ax.grid(axis="y")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}") if ax != ax_q else StrMethodFormatter("{x:g}"))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def cartpole_chart(path, grid):
    angles = np.unique(grid["angle_rad"])
    speeds = np.unique(grid["angular_velocity_rad_s"])
    fig, axes = plt.subplots(1, 2, figsize=(7.13, 2.63), layout="constrained", sharey=True)
    for ax, column, title in zip(axes, ("q_right_minus_left", "return_right_minus_left"),
                                 ("Network action values", "Forced first action rollouts")):
        values = grid[column].reshape(len(angles), len(speeds))
        limit = max(np.abs(values).max(), 1e-6)
        mesh = ax.pcolormesh(speeds, angles, values, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest")
        ax.axvline(0, color=INK, alpha=0.33, lw=0.6)
        ax.axhline(0, color=INK, alpha=0.33, lw=0.6)
        ax.set(title=title, xlabel="Pole angular velocity (rad/s)")
        fig.colorbar(mesh, ax=ax, shrink=0.86, label="Right minus left", pad=0.035)
    axes[0].set_ylabel("Pole angle (rad)")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def atari_chart(path, trace):
    fig, axes = plt.subplots(2, 1, figsize=(7.13, 2.7), layout="constrained", sharex=True)
    axes[0].plot(trace["step"], trace["max_q"], color=BLUE, lw=1.0)
    axes[1].step(trace["step"], trace["score"], color=BLUE, where="post", lw=1.1)
    for ax in axes:
        for i, event in enumerate(trace[trace["life_lost"] == 1]):
            ax.axvline(event["step"], lw=1, ls="--", color=ORANGE, alpha=0.8,
                       label="Life lost" if i == 0 else None)
        ax.grid(axis="y")
        ax.yaxis.set_major_locator(MaxNLocator(4))
    if np.any(trace["life_lost"]):
        axes[0].legend(loc="upper right")
    axes[0].set(ylabel="Max Q", title="Greedy Q estimate and score within one game")
    axes[1].set(xlabel="Agent step (four emulator frames)", ylabel="Raw score")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def styles():
    directory = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("ReportSans", str(directory / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("ReportSansBold", str(directory / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSansBold")
    return {
        "title": ParagraphStyle("Title", fontName="ReportSansBold", fontSize=29, leading=34,
                                textColor=INK, spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", fontName="ReportSans", fontSize=11, leading=16,
                                   textColor=MUTED, spaceAfter=14),
        "heading": ParagraphStyle("Heading", fontName="ReportSansBold", fontSize=13, leading=18,
                                  textColor=INK, spaceBefore=11, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("Body", fontName="ReportSans", fontSize=9.5, leading=14.1,
                               textColor=INK, spaceAfter=7),
        "caption": ParagraphStyle("Caption", fontName="ReportSans", fontSize=8.7, leading=12.5,
                                  textColor=MUTED, spaceAfter=5),
        "cell": ParagraphStyle("Cell", fontName="ReportSans", fontSize=8.7, leading=12, textColor=INK),
        "header_cell": ParagraphStyle("HeaderCell", fontName="ReportSansBold", fontSize=8.6, leading=12,
                                      textColor=colors.white),
        "equation": ParagraphStyle("Equation", fontName="ReportSans", fontSize=10, leading=15,
                                   alignment=TA_CENTER, backColor=PALE, borderPadding=7, spaceBefore=6,
                                   spaceAfter=10, textColor=INK),
        "code": ParagraphStyle("Code", fontName="Courier", fontSize=8.2, leading=12,
                               backColor=PALE, borderPadding=7, spaceBefore=4, spaceAfter=8, textColor=INK),
    }


def frame(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor(LIGHT))
    canvas.line(51, 743, 561, 743)
    canvas.line(51, 47, 561, 47)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.setFont("ReportSans", 8)
    canvas.drawString(51, 753, "MLDS 490  /  ASSIGNMENT 2")
    canvas.drawRightString(561, 753, "DEEP Q-NETWORK")
    canvas.drawString(51, 34, "SAMSUNYUJUN  /  mlds-rl-assignment2")
    canvas.drawRightString(561, 34, str(doc.page))
    canvas.restoreState()


def generate(cart_dir, ms_dir, output):
    cart = evidence(cart_dir)
    ms = evidence(ms_dir)
    theme = styles()
    chart_style()
    story = []

    def paragraph(text, style="body"):
        story.append(Paragraph(text, theme[style]))

    def table(rows, widths):
        content = [[Paragraph(str(cell), theme["header_cell" if i == 0 else "cell"]) for cell in row]
                   for i, row in enumerate(rows)]
        element = Table(content, colWidths=widths, hAlign="LEFT", repeatRows=1)
        element.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(PALE)]),
            ("LINEBELOW", (0, 1), (-1, -1), 0.35, colors.HexColor(LIGHT)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([element, Spacer(1, 7)])

    def page(title, subtitle):
        if story:
            story.append(PageBreak())
        paragraph(title, "title")
        paragraph(subtitle, "subtitle")

    def image(path, height):
        story.append(Image(str(path), width=WIDTH, height=height))

    def policy_table(validation, label, names):
        rows = [["Policy", f"Mean {label}", "Selected minus policy (95% paired CI)"]]
        for key, display in names:
            contrast = validation["selected_minus"].get(key)
            interval = "Reference policy"
            if contrast:
                interval = f"{contrast['delta']:+,.2f} [{contrast['ci_low']:+,.2f}, {contrast['ci_high']:+,.2f}]"
            rows.append([display, f"{validation['means'][key]:,.2f}", interval])
        table(rows, [190, 95, 225])

    with TemporaryDirectory(prefix="mlds-a2-plots-") as directory:
        figures = Path(directory)
        episode_charts(figures / "cartpole.png", cart)
        episode_charts(figures / "mspacman.png", ms)
        cartpole_chart(figures / "cartpole_grid.png", read_csv(cart_dir / "validation_grid.csv"))
        atari_chart(figures / "atari_trace.png", read_csv(ms_dir / "validation_trace.csv"))

        page("Deep Q-Network", "CartPole-v1 and ALE/MsPacman-v5")
        paragraph(f'<b>GitHub</b> SAMSUNYUJUN / mlds-rl-assignment2<br/>'
                  f'<link href="{REPOSITORY}" color="{BLUE}">{REPOSITORY}</link>')
        result_rows = [["Environment", "Training episodes", "Selected checkpoint", "500-episode mean ± SD"]]
        for name, (_, _, meta, result, _) in (("CartPole-v1", cart), ("ALE/MsPacman-v5", ms)):
            result_rows.append([name, f"{meta['episodes']:,}", f"{result['checkpoint_episode']:,}",
                                f"{result['mean']:,.2f} ± {result['sd']:,.2f}"])
        table(result_rows, [150, 100, 105, 155])
        paragraph("Both agents used an online action-value network, a uniformly sampled replay buffer, and a lagged "
                  "target network. The experiments used a single NVIDIA RTX A6000 on DeepDish." if
                  all(data[2]["device"].startswith("cuda") for data in (cart, ms)) else
                  "Both agents used an online action-value network, a uniformly sampled replay buffer, and a lagged "
                  "target network. The device used for each run appears in the environment sections.")

        paragraph("Learning rule", "heading")
        paragraph("The network combines a state-value head with an action-advantage head after subtracting the mean "
                  "advantage. At each update the online network selects the next action and the target network evaluates "
                  "it. Adam minimizes the Huber loss between Q(s, a) and the following target.")
        paragraph("y = Σ<sub>j=0</sub><super>k−1</super> γ<super>j</super>r<sub>j</sub> + "
                  "d Q<sub>target</sub>(s′, argmax<sub>a′</sub> Q<sub>online</sub>(s′, a′))", "equation")
        paragraph("The multiplier d is γ<super>k</super> while continuation is possible and 0 at a learning terminal. "
                  "Each lane has its own queue of up to three transitions. The learner drains shorter tails at boundaries, "
                  "so the target never includes a reward from the following episode or life.")
        paragraph("Exploration starts with uniform actions. Updates begin after a replay warmup, and ε then decreases "
                  "linearly with agent steps. Replay samples only filled slots and overwrites the oldest entries when full. "
                  "The learner excludes Gymnasium's reset-only steps from replay, Q statistics, and episode returns.")

        paragraph("Evaluation protocol", "heading")
        paragraph("Periodic greedy rollouts on fixed selection seeds choose the checkpoint with the highest mean raw reward. "
                  "Ties keep the earlier checkpoint. Each final evaluation uses ε = 0 and 25 newly seeded environment "
                  "streams with 20 episodes per stream. The tables and histograms use raw rewards and population SD. "
                  "Policy probes use a third seed range.")
        paragraph("The Q plots show the largest estimated Q over all actions and visited states before actions in each "
                  "training episode. The 100-episode moving averages begin at episode 100. The orange marker locates the "
                  "selected checkpoint.")

        page("CartPole-v1", "Training and 500 greedy evaluation episodes")
        train, _, meta, summary, val = cart
        cfg = meta["config"]
        scaling = "Input scales are 2.4, 2, π/15 and 2 in the environment's observation order. " if cfg.get("normalize_cartpole") else ""
        paragraph(f"The agent observes cart position, cart velocity, pole angle and pole angular velocity and chooses left "
                  f"or right. {scaling}A 128-unit encoder feeds two 128-unit heads. The discount is 0.95.")
        image(figures / "cartpole.png", 438)
        paragraph(f"<b>Evaluation.</b> Mean {summary['mean']:,.2f}, SD {summary['sd']:,.2f}, range "
                  f"{summary['min']:,.0f} to {summary['max']:,.0f}. The pole remained upright through the 500-step limit in "
                  f"{summary['balanced_to_500']}/500 episodes. Checkpoint {summary['checkpoint_episode']:,} was selected "
                  f"using {cfg['validation_episodes']} evaluation seeds.")
        paragraph(f"The first and final 100 training episodes averaged {train['reward'][:100].mean():.2f} and "
                  f"{train['reward'][-100:].mean():.2f}. The final frozen policy averaged {val['means']['last']:.2f} in "
                  "the paired probe, which shows that continued updates can degrade a policy. At γ = 0.95, even 150 "
                  "steps already yield nearly 20 discounted reward, so high Q alone gives little evidence of 500-step "
                  "balance.", "caption")

        page("CartPole validation", "First-action interventions and the 500-step boundary")
        grid = val["grid"]
        boundary = val["boundaries"]
        image(figures / "cartpole_grid.png", 187)
        paragraph(f"<b>Figure.</b> For each of {grid['grid_states']} fixed pole states with the cart centered and still, "
                  "the experiment forces left or right once, then runs the same greedy controller. Red favors right, "
                  "blue favors left. The right panel uses realized discounted return through termination or 500 steps.", "caption")
        paragraph(f"Among the {grid['informative_states']} states with an absolute return gap of at least 0.10, the "
                  f"network preferred the better first action in {grid['action_agreement']}. Following the predicted first "
                  f"action lost {grid['mean_regret']:.6f} discounted reward on average across all states, with worst loss "
                  f"{grid['worst_regret']:.4f}. This grid includes angles and velocities beyond the normal reset distribution.")
        policy_table(val, "steps", [("random", "Uniform random"), ("initial", "Untrained greedy"),
                                    ("last", f"Final checkpoint ({val['last_episode']:,})"), ("selected", "Selected greedy")])
        paragraph(f"The comparison uses {val['episodes_per_policy']} common reset seeds. The intervals resample paired "
                  "seed differences 10,000 times.", "caption")
        paragraph("Physical failure and the time cap", "heading")
        paragraph(f"A seeded three-step rollout and an otherwise identical capped rollout reached the same final state. "
                  f"The cap kept the continuation term, giving a one-step target of {boundary['target_at_cap']:.3f}. "
                  f"Treating the cap as physical failure would give {boundary['target_if_mislabeled_terminal']:.1f}. "
                  f"A forced angle failure from θ = 0.208 rad and angular velocity 0.4 rad/s produced target "
                  f"{boundary['failure_target']:.1f} with zero continuation. Physical failure takes precedence when both "
                  "end flags appear at step 500.")

        page("MsPacman-v5", "Training and 500 greedy evaluation episodes")
        train, _, meta, summary, val = ms
        cfg = meta["config"]
        paragraph("ALE supplies four stacked 84 × 84 grayscale observations with max pooling. The CNN uses three "
                  "convolutions and two 128-unit heads to score nine actions. The v5 emulator retains four-frame action "
                  "repeat and 25% sticky actions. The discount is 0.99 and replay uses raw points divided by 10.")
        image(figures / "mspacman.png", 438)
        paragraph(f"<b>Evaluation.</b> Mean {summary['mean']:,.2f}, SD {summary['sd']:,.2f}, median "
                  f"{summary['median']:,.0f} and range {summary['min']:,.0f} to {summary['max']:,.0f}. The selected "
                  f"checkpoint followed episode {summary['checkpoint_episode']:,}.")
        paragraph(f"The first and final 100 training episodes averaged {train['reward'][:100].mean():,.2f} and "
                  f"{train['reward'][-100:].mean():,.2f} raw points. The peak Q uses scaled, discounted rewards and "
                  "per-life bootstrap boundaries. The raw score curves and fixed-seed greedy comparisons measure policy "
                  "improvement directly.", "caption")

        page("MsPacman validation", "Motion information, score progress and life boundaries")
        image(figures / "atari_trace.png", 192)
        paragraph(f"<b>Figure.</b> The selected policy on seed {val['trace_seed']:,}. Orange marks life losses. The Q "
                  "trace gives the pre-action estimate and the score trace includes the reward earned on that action.", "caption")
        policy_table(val, "points", [("random", "Uniform random"), ("initial", "Untrained greedy"),
                                     ("early", f"Early checkpoint ({val['early_episode']:,})"),
                                     ("last", f"Final checkpoint ({val['last_episode']:,})"),
                                     ("frozen_frames", "Selected with repeated newest frame"),
                                     ("selected", "Selected with frame history")])
        paragraph(f"The {val['episodes_per_policy']} seed pairs use identical reset seeds and v5 settings. Repeating the "
                  "newest frame removes the temporal differences the trained CNN receives. The intervention also changes "
                  "the input distribution. Retraining with a single-frame input would isolate the benefit of frame history.", "caption")
        paragraph("Life loss, game over and the frame cap", "heading")
        game_events = [x for x in val["boundaries"] if x["kind"] == "life loss"]
        first_loss = f" at action {game_events[0]['step']:,}" if game_events else ""
        cap = val["forced_cap"]
        d = ", ".join(f"{x:.6g}" for x in cap["bootstrap_multipliers"])
        g = ", ".join(f"{x:.6g}" for x in cap["returns"])
        paragraph(f"The replay tail drains with zero bootstrap at a lost life{first_loss}. The game continues with the "
                  "same score accumulator. In the recorded trajectory, the replay returns match an independent backward "
                  f"calculation at all {len(val['boundaries'])} life and episode boundaries. On the same trajectory capped "
                  f"at {cap['frame_limit']:,} emulator frames, the replay tail retained returns ({g}) and bootstrap "
                  f"multipliers ({d}). Reset-only steps are excluded from the traces and replay.")

        page("Reproduction", "Commands, parameter settings and further validation")
        paragraph("Run the following commands from a DeepDish login. Set A2_GPU to the assigned GPU index. The script "
                  "trains both policies, evaluates 500 episodes per policy, and saves all validation data. Use a fresh runs "
                  "directory for each full run.")
        story.append(Preformatted(
            "git clone https://github.com/SAMSUNYUJUN/mlds-rl-assignment2.git\n"
            "cd mlds-rl-assignment2\n"
            "python3 -m venv --system-site-packages .venv\n"
            ".venv/bin/python -m pip install -r requirements.txt\n"
            "A2_GPU=0 bash deepdish.sh\n"
            ".venv/bin/python make_report.py --cartpole runs/cartpole --mspacman runs/mspacman",
            theme["code"],
        ))
        rows = [["Setting", "CartPole-v1", "ALE/MsPacman-v5"]]
        for label, key in (("Training seed", "seed"), ("Parallel environments", "num_envs"),
                           ("Replay capacity / warmup", None), ("Batch size / update interval", None),
                           ("Adam learning rate", "learning_rate"), ("Target copy (updates)", "target_every"),
                           ("ε decay (steps) / floor", None), ("N-step horizon", "n_step")):
            values = []
            for data in (cart, ms):
                cfg = data[2]["config"]
                paired = {"Replay capacity / warmup": ("capacity", "learning_starts"),
                          "Batch size / update interval": ("batch_size", "train_every"),
                          "ε decay (steps) / floor": ("epsilon_steps", "epsilon_end")}
                if key:
                    values.append(f"{cfg[key]:,}")
                else:
                    first, second = paired[label]
                    values.append(f"{cfg[first]:,} / {cfg[second]:,}")
            rows.append([label, *values])
        table(rows, [244, 133, 133])
        paragraph(f"CartPole ran {cart[2]['steps']:,} agent steps with PyTorch {cart[2]['torch']} on {cart[2]['hardware']}. "
                  f"MsPacman ran {ms[2]['steps']:,} agent steps with PyTorch {ms[2]['torch']} on {ms[2]['hardware']}. "
                  "README.md documents individual commands and the recorded files.", "caption")
        paragraph("Experiments with three more months", "heading")
        paragraph("<b>Month 1.</b> Repeat training over at least ten seeds, compare one-step and three-step targets at "
                  "matched environment steps, and report between-seed variation in time to solve CartPole. Sweep cart "
                  "position and velocity as well as pole state, then test sustained balance beyond the nominal cap.")
        paragraph("<b>Month 2.</b> Retrain MsPacman with one versus four frames, with and without life-loss bootstrapping, "
                  "under equal frame and update budgets. Track score per life, maze clears, time to first death and "
                  "the median score alongside the mean to measure the effect of rare bonus rewards.")
        paragraph("<b>Month 3.</b> Use fresh seed sets and vary stickiness and initial no-op counts. For both games, "
                  "compare predicted Q to empirical discounted returns from cloned states under each available first action. "
                  "Report value bias, action regret and bootstrap intervals over independent training seeds.")
        paragraph(f'<link href="https://doi.org/10.1038/nature14236" color="{BLUE}">Mnih et al. (2015), DQN</link> · '
                  f'<link href="https://doi.org/10.1609/aaai.v30i1.10295" color="{BLUE}">van Hasselt et al. (2016), Double DQN</link> · '
                  f'<link href="https://proceedings.mlr.press/v48/wangf16.html" color="{BLUE}">Wang et al. (2016), dueling networks</link><br/>'
                  f'<link href="https://gymnasium.farama.org/environments/classic_control/cart_pole/" color="{BLUE}">CartPole environment</link> · '
                  f'<link href="https://ale.farama.org/environments/ms_pacman/" color="{BLUE}">ALE MsPacman environment</link>', "caption")

        output.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(output), pagesize=letter, leftMargin=45, rightMargin=45,
                                topMargin=57, bottomMargin=52, title="Assignment 2 - Deep Q-Network", author="Yujun Sun")
        doc.build(story, onFirstPage=frame, onLaterPages=frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cartpole", required=True, type=Path)
    parser.add_argument("--mspacman", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/pdf/assignment2_report.pdf"))
    args = parser.parse_args()
    generate(args.cartpole, args.mspacman, args.out)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
