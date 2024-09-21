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
        p = re.compile("([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")
        with open_file(filename) as f:
            for line in f:
                m = p.match(line)
                if not m:
                    print("---------------------> IGNORING : ", line)
                else:
                    fen, bm = m.group(1), int(m.group(2))
                    plies = 2 * bm - 1 if bm > 0 else -2 * bm
                    self.plies[plies] += 1
        self.filename = filename[:-3] if filename.endswith(".gz") else filename
        print(
            f"Loaded {sum(self.plies.values())} EPDs with abs(bm) in [{(min(self.plies.keys()) + 1) // 2}, {(max(self.plies.keys()) + 1) // 2}]."
        )
        s = sum(key * count for key, count in self.plies.items())
        l = sum(self.plies.values())
        if l:
            print(f"Average length of bm mates in plies is {s/l:.2f}.")
        if debug:
            print("ply frequencies:", end=" ")
            ply_count = sorted(self.plies.items(), key=lambda x: x[0])
            print(", ".join([f"{ply}: {frequency}" for ply, frequency in ply_count]))

    def create_graph(self, bucketSize=1, cutOff=200):
        plies = Counter()
        for p, freq in self.plies.items():
            plies[min(p, cutOff)] += freq
        rangeMin, rangeMax = min(plies.keys()), max(plies.keys())
        fig, ax = plt.subplots()
        ax.hist(
            plies.keys(),
            weights=plies.values(),
            range=(rangeMin, rangeMax),
            bins=(rangeMax - rangeMin) // bucketSize,
            density=False,
            alpha=0.5,
            color="blue",
            edgecolor="black",
            align="left",
        )
        for patch in ax.patches:
            bin_x = patch.get_x() + patch.get_width() / 2
            if int(bin_x) % 2 == 0:
                patch.set_facecolor("deepskyblue")
        pos = mpatches.Patch(color="blue", label="bm > 0")
        neg = mpatches.Patch(color="deepskyblue", label="bm < 0")
        ax.legend(handles=[pos, neg])
        ax.set_xlabel("bm")
        fig.suptitle(
            f"bm distribution for {self.filename}.",
        )
        if max(self.plies.keys()) > cutOff:
            ax.set_title(
                f"(Ply values > {cutOff} are included in the {cutOff} bucket.)",
                fontsize=6,
                family="monospace",
            )
        xticks = [int(x) + 1 for x in ax.get_xticks()]
        xticks = [x for x in xticks if x >= rangeMin and x <= rangeMax and x % 2]
        new_xtick_labels = [(x + 1) // 2 for x in xticks]
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
        help="Cutoff value for the distribution plot (in plies).",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-b",
        "--bucket",
        type=int,
        default=1,
        help="bucket size for evals",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show frequency data on stdout.",
    )
    args = parser.parse_args()

    d = data(args.filename, args.debug)
    d.create_graph(args.bucket, args.cutOff)
