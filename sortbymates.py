import argparse, re


def sort_key(t, stable, multiPV):
    _, bm, _ = t
    if multiPV:
        key = (1, 0) if bm is None else (0, bm) if bm > 0 else (2, bm)
        return key if stable else (*key, t[0] or "")
    key = float("inf") if bm is None else abs(bm) + (0.5 if bm < 0 else 0)
    return key if stable else (key, t[0] or "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort the mate puzzles in ascending order, preserving comment boundaries. Output is to stdout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="The .epd file to be sorted.")
    parser.add_argument(
        "--stable",
        action="store_true",
        help="Use stable sort; default sorts same-mate entries by FEN.",
    )
    parser.add_argument(
        "--multiPV",
        action="store_true",
        help="Sort by quality: winning mates (shortest first), unknown, losing mates (longest first).",
    )
    args = parser.parse_args()
    p = re.compile(r"^([1-8a-zA-Z/]+ [wb] [a-zA-Z\-]+ [a-h1-8\-]+)( bm #(-?\d+);)?")

    segments = []  # list of (comment_or_None, [fens_upto_comment])
    current_fens = []

    with open(args.source) as f:
        for line in f:
            if line.startswith("#"):
                segments.append((None, current_fens))
                current_fens = []
                segments.append((line, []))
            else:
                m = p.match(line)
                assert m, f"error for line '{line}' in file {args.source}"
                fen = m.group(1)
                bm = int(m.group(3)) if m.group(2) is not None else None
                current_fens.append((fen, bm, line))

    segments.append((None, current_fens))

    for header, fens in segments:
        if header is not None:
            print(header, end="")
        if fens:
            fens.sort(key=lambda t: sort_key(t, args.stable, args.multiPV))
            for _, _, line in fens:
                print(line, end="")
