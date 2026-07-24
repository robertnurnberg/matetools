import argparse, chess, gzip, re


def open_file(filename):
    open_func = gzip.open if filename.endswith(".gz") else open
    return open_func(filename, "rt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A diff-like script to show the differences between two Chest-like .epd(.gz) files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file1")
    parser.add_argument("file2")
    args = parser.parse_args()
    p = re.compile(r"^([1-8a-zA-Z/]+ [wb] [a-zA-Z\-]+ [a-h1-8\-]+)( bm #(-?\d+);)?")

    d = [{}, {}]
    for idx, filename in enumerate([args.file1, args.file2]):
        with open_file(filename) as f:
            for line in f:
                if line.startswith("#"):  # ignore comments
                    continue
                m = p.match(line)
                assert m, f"error for line '{line[:-1]}' in file {filename}"
                fen = m.group(1)
                bm = int(m.group(3)) if m.group(2) is not None else None
                assert fen not in d[idx], f'error: duplicate FEN "{fen}" in {filename}'
                d[idx][fen] = bm, line

    for fen, (bm1, line1) in d[0].items():
        bm2, line2 = d[1].get(fen, (None, None))
        if line2 is None:
            print("> " + line1 + "<\n---")
        elif bm2 != bm1:
            print("> " + line1 + "< " + line2 + "---")

    for fen, (_, line2) in d[1].items():
        _, line1 = d[0].get(fen, (None, None))
        if line1 is None:
            print(">\n< " + line2 + "---")
