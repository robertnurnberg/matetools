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
    p = re.compile(r"^([1-8a-zA-Z/]+ [wb] [a-zA-Z\-]+ [a-h1-8\-]+)( bm #(-?\d+);)?")

    fens = []
    with open(args.source) as f:
        for line in f:
            if line.startswith("#"):
                continue
            m = p.match(line)
            assert m, f"error for line '{line[:-1]}' in file {args.source}"
            fen = m.group(1)
            bm = int(m.group(3)) if m.group(2) is not None else None
            if bm is None:
                continue
            fens.append((fen, bm, line))

    if args.stable:
        fens.sort(key=lambda t: abs(t[1]) + (0.5 if t[1] < 0 else 0))
    else:
        fens.sort(key=lambda t: (abs(t[1]) + (0.5 if t[1] < 0 else 0), t[0]))
    for _, _, line in fens:
        print(line[:-1])
