import argparse, gzip, re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter


def open_file(filename):
    open_func = gzip.open if filename.endswith(".gz") else open
    return open_func(filename, "rt")


class data:
    def __init__(self, filename, debug=False):
        self.plies = Counter()
        p = re.compile(r"^([1-8a-zA-Z/]+ [wb] [a-zA-Z\-]+ [a-h1-8\-]+)( bm #(-?\d+);)?")
        loaded = set()
        self.bmplus = self.bmminus = 0
        with open_file(filename) as f:
            for line in f:
                if line.startswith("#"):  # ignore comments
                    continue
                m = p.match(line)
                if not m:
                    print("---------------------> IGNORING : ", line)
                    continue
                fen = m.group(1)
                bm = int(m.group(3)) if m.group(2) is not None else None
                if fen in loaded:
                    print(f"Warning: Found duplicate FEN {fen}.")
                    continue
                loaded.add(fen)
                if bm:
                    plies = 2 * bm - 1 if bm > 0 else -2 * bm
                    self.plies[plies] += 1
                    if bm > 0:
                        self.bmplus += 1
                    else:
                        self.bmminus += 1
        self.filename = filename[:-3] if filename.endswith(".gz") else filename
        totalbm = self.bmplus + self.bmminus
        self.bmmin = ((min(self.plies.keys()) + 1) // 2) if totalbm else 0
        self.bmmax = ((max(self.plies.keys()) + 1) // 2) if totalbm else 0
        print(
            f"Loaded {len(loaded)} unique EPDs with {totalbm} bm values, with |bm| in [{self.bmmin}, {self.bmmax}]."
        )
        if not totalbm:
            return
        s = sum((key + 1) // 2 * count for key, count in self.plies.items())
        print(f"Average for |bm| is {s/totalbm:.2f}.")
        if debug:
            print("bm frequencies:", end=" ")
            ply_count = sorted(self.plies.items(), key=lambda x: x[0])
            print(
                ", ".join(
                    [
                        f"#{(ply + 1) // 2 if ply % 2 else - ply // 2}: {frequency}"
                        for ply, frequency in ply_count
                    ]
                )
            )

    def create_graph(self, cutOff):
        totalbm = self.bmplus + self.bmminus
        if not totalbm:
            return
        plies = Counter()
        for p, freq in self.plies.items():
            if p > 2 * cutOff:
                plies[2 * cutOff - 1 if p % 2 == 1 else 2 * cutOff] += freq
            else:
                plies[p] += freq
        rangeMin, rangeMax = min(plies.keys()), max(plies.keys())
        fig, ax = plt.subplots()
        ax.hist(
            plies.keys(),
            weights=plies.values(),
            range=(rangeMin, rangeMax + 1),
            bins=rangeMax + 1 - rangeMin,
            density=False,
            alpha=0.5,
            color="blue",
            edgecolor="black",
        )
        for patch in ax.patches:
            bin_x = patch.get_x() + patch.get_width() / 2
            if int(bin_x) % 2 == 0:
                patch.set_facecolor("deepskyblue")
        pos = mpatches.Patch(color="blue", label=f"bm > 0 (total: {self.bmplus})")
        neg = mpatches.Patch(
            color="deepskyblue", label=f"bm < 0 (total: {self.bmminus})"
        )
        ax.legend(handles=[pos, neg])
        ax.set_xlabel("|bm|")
        fig.suptitle(
            f"Distribution plot for the {totalbm} bm's in {self.filename}.",
        )
        if max(self.plies.keys()) > 2 * cutOff:
            ax.set_title(
                f"Values |bm| > {cutOff} are included in the {cutOff} buckets.",
                fontsize=6,
                family="monospace",
            )
        xticks = (
            [(rangeMin + 1) // 2 * 2]
            + list(ax.get_xticks())
            + [(rangeMax + 1) // 2 * 2]
        )
        xticks = [
            int(x) for x in xticks if x >= rangeMin and x <= rangeMax and x % 2 == 0
        ]
        new_xtick_labels = [x // 2 for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(new_xtick_labels)
        prefix, _, _ = self.filename.rpartition(".")
        pgnname = prefix + ".png"
        plt.savefig(pgnname, dpi=300)
        print(f"Saved bm distribution plot in file {pgnname}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot bm distribution for positions in e.g. matetrack.epd.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        help=".epd(.gz) file with positions and their cdb evals.",
    )
    parser.add_argument(
        "-c",
        "--cutOff",
        help="Cutoff value for the distribution plot.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show frequency data on stdout.",
    )
    args = parser.parse_args()

    d = data(args.filename, args.debug)
    d.create_graph(args.cutOff)
