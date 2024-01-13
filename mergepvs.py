import argparse, re, chess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge several .epd files with mates and PVs, which are assumed to be correct (possibly too short).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="The source .epd file.")
    parser.add_argument(
        "references",
        nargs="*",
        help="List of .epd files with possibly more or longer PVs.",
    )
    args = parser.parse_args()
    p = re.compile("([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")

    d = {}
    for filename in [args.source] + args.references:
        with open(filename) as f:
            for line in f:
                m = p.match(line)
                assert m, "error"
                fen, bm = m.group(1), int(m.group(2))
                _, _, pv = line.partition("; PV: ")
                pv, _, _ = pv[:-1].partition(";")  # remove '\n'
                pv = pv.split()
                bmold, pvold = d.get(fen, (None, None))
                if (
                    bmold is None
                    or abs(bm) < abs(bmold)
                    or bm == bmold
                    and len(pv) > len(pvold)
                ):
                    d[fen] = bm, pv

    with open(args.source) as f:
        for line in f:
            m = p.match(line)
            assert m, "error"
            fen, bm = m.group(1), int(m.group(2))
            bm, pv = d.get(fen, (0, None))
            if pv is not None and pv:
                print(f"{fen} bm #{bm}; PV: {' '.join(pv)};")
            else:
                print(line[:-1])
