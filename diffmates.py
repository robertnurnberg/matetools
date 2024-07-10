import argparse, chess, re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A diff-like script to show the differences between two Chest-like .epd files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file1")
    parser.add_argument("file2")
    args = parser.parse_args()
    p = re.compile("([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")

    d = [{}, {}]
    for idx, filename in enumerate([args.file1, args.file2]):
        with open(filename) as f:
            for line in f:
                m = p.match(line)
                assert m, f"error for line '{line[:-1]}' in file {filename}"
                fen, bm = m.group(1), int(m.group(2))
                assert fen not in d[idx], f'error: duplicate FEN "{fen}" in {filename}'
                d[idx][fen] = bm, line

    for fen, (bm1, line1) in d[0].items():
        bm2, line2 = d[1].get(fen, (None, None))
        if bm2 is None:
            print("> " + line1 + "<\n---")
        elif bm2 != bm1:
            print("> " + line1 + "< " + line2 + "---")

    for fen, (_, line2) in d[1].items():
        bm1, _ = d[0].get(fen, (None, None))
        if bm1 is None:
            print(">\n< " + line2 + "---")
