import argparse, chess, re


def add_tuple(d, fen, bm, pv):
    bmold, pvold = d.get(fen, (None, None))
    if bmold is None or abs(bm) < abs(bmold) or bm == bmold and len(pv) > len(pvold):
        d[fen] = bm, pv.copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uses proven PVs, and the associated PVs for all the positions along the mating lines, to find possibly missing PVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epdFile",
        default="matetrackpv.epd",
        help="file containing the positions, their mate scores and their PVs",
    )
    parser.add_argument(
        "references",
        nargs="+",
        help="List of .epd files with proven PVs.",
    )
    parser.add_argument(
        "--outFile",
        default="deduced.epd",
        help="output file with deduced positions, their mate scores and PVs",
    )
    args = parser.parse_args()
    p = re.compile(r"([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")

    fencount = 0
    d = {}  # the dict will hold the shortest mates, with longest PVs
    for filename in args.references:
        with open(filename) as f:
            for line in f:
                m = p.match(line)
                assert m, f"error for line '{line[:-1]}' in file {filename}"
                fen, bm = m.group(1), int(m.group(2))
                _, _, pv = line.partition("; PV: ")
                pv, _, _ = pv[:-1].partition(";")  # remove '\n'
                pv = pv.split()
                add_tuple(d, fen, bm, pv)
                board = chess.Board(fen)
                while pv:
                    move = pv.pop(0)
                    board.push(chess.Move.from_uci(move))
                    if pv == [] and not bool(board.legal_moves):
                        break
                    bm = -bm + (1 if bm > 0 else 0)
                    fen = board.epd()
                    add_tuple(d, fen, bm, pv)
                fencount += 1

    print(f"Deduced {len(d.keys())} mating PVs from {fencount} given FENs.")

    count = 0
    with open(args.epdFile) as fin, open(args.outFile, "w") as fout:
        for line in fin:
            m = p.match(line)
            assert m, f"error for line '{line[:-1]}' in file {args.epdFile}"
            fen, bm = m.group(1), int(m.group(2))
            _, _, pv = line.partition("; PV: ")
            pv, _, _ = pv[:-1].partition(";")  # remove '\n'
            pv = pv.split()
            bmnew, pvnew = d.get(fen, (None, None))
            if (
                bmnew is None
                or abs(bmnew) > abs(bm)
                or bmnew == bm
                and len(pvnew) <= len(pv)
            ):
                fout.write(line)
            else:
                fout.write(f"{fen} bm #{bmnew}; PV: {' '.join(pvnew)};\n")
                count += 1

    print(f"Found/lengthened {count} PVs.")
