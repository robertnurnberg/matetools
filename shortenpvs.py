import argparse, chess, random, re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shorten existing PVs by removing moves from the end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epdFile",
        default="matetrackpv.epd",
        help="file containing the positions, their mate scores and their PVs",
    )
    parser.add_argument(
        "--outFile",
        default="shortpvs.epd",
        help="output file with shortened PVs",
    )
    parser.add_argument(
        "--plies",
        type=int,
        default=1,
        help="number of plies to remove, if possible",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="remove random number X of plies, with X in [0, PLIES].",
    )
    args = parser.parse_args()

    p = re.compile("([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")

    fens = count = 0
    with open(args.epdFile) as fin, open(args.outFile, "w") as fout:
        for line in fin:
            m = p.match(line)
            assert m, f"error for line '{line[:-1]}' in file {args.epdFile}"
            fen, bm = m.group(1), int(m.group(2))
            _, _, pv = line.partition("; PV: ")
            pv, _, _ = pv[:-1].partition(";")  # remove '\n'
            pv = pv.split()
            plies = random.randint(0, args.plies) if args.random else args.plies
            if plies == 0 or plies >= len(pv):
                fout.write(line)
            else:
                fout.write(f"{fen} bm #{bm}; PV: {' '.join(pv[:-plies])};\n")
                count += 1
            fens += 1

    print(f"Loaded {fens} FENs, shortened {count} PVs.")
