import argparse, chess, re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort the mate puzzles in ascending order. Output is to stdout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="The .epd file to be sorted.")
    parser.add_argument(
        "--stable",
        action="store_true",
        help="Use stable sort, default is sort same mates by FEN.",
    )
    args = parser.parse_args()
    p = re.compile("([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")

    fens = []
    with open(args.source) as f:
        for line in f:
            m = p.match(line)
            assert m, "error"
            fen, bm = m.group(1), int(m.group(2))
            fens.append((fen, bm, line))

    if args.stable:
        fens.sort(key=lambda t: abs(t[1]) + (0.5 if t[1] < 0 else 0))
    else:
        fens.sort(key=lambda t: (abs(t[1]), t[0]))
    for _, _, line in fens:
        print(line[:-1])
