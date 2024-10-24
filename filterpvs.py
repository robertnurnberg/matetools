import argparse, chess, gzip, re


def open_file(filename):
    open_func = gzip.open if filename.endswith(".gz") else open
    return open_func(filename, "rt")


def pv_status(fen, mate, pv):
    # check if the given pv (list of uci moves) leads to checkmate #mate
    losing_side = 1 if mate > 0 else 0
    try:
        board = chess.Board(fen)
        for ply, move in enumerate(pv):
            if ply % 2 == losing_side and board.can_claim_draw():
                return "draw"
            uci = chess.Move.from_uci(move)
            if not uci in board.legal_moves:
                raise Exception(f"illegal move {move} at position {board.epd()}")
            board.push(uci)
    except Exception as ex:
        return f'error "{ex}"'
    plies_to_checkmate = 2 * mate - 1 if mate > 0 else -2 * mate
    if len(pv) < plies_to_checkmate:
        return "short"
    if len(pv) > plies_to_checkmate:
        return "long"
    if board.is_checkmate():
        return "ok"
    return "wrong"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter positions in a .epd(.gz) file by the status of their mate PVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epdFile",
        default="matetrackpv.epd",
        help="File containing positions, their mate scores and PVs.",
    )
    parser.add_argument(
        "--outFile",
        default="filteredpvs.epd",
        help="Output file for the filtered positions and their mate PVs.",
    )
    parser.add_argument(
        "--status",
        default="ok",
        help="Filter the PVs by status: ok, short, long, draw, wrong, all.",
    )
    args = parser.parse_args()

    p = re.compile("([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")

    filtered = []
    allowed = args.status.split("+")
    with open_file(args.epdFile) as f:
        for line in f:
            m = p.match(line)
            assert m, f"error for line '{line[:-1]}' in file {args.epdFile}"
            fen, bm = m.group(1), int(m.group(2))
            _, _, pv = line.partition("; PV: ")
            pv, _, _ = pv[:-1].partition(";")  # remove '\n'
            pv = pv.split()
            if pv:
                status = pv_status(fen, bm, pv)
                if "all" in allowed or status in allowed:
                    filtered.append(line)

    if filtered:
        with open(args.outFile, "w") as f:
            for line in filtered:
                f.write(line)
        print(f"All done. Saved {len(filtered)} PVs to {args.outFile}.")
    else:
        print(f"All done. No PVs matching status '{args.status}' found.")
