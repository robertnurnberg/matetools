import argparse, chess, chess.engine, os, re


def pv_status(fen, mate, pv):
    # check if the given pv (list of uci moves) leads to checkmate #mate
    plies_to_checkmate = 2 * mate - 1 if mate > 0 else -2 * mate
    if len(pv) < plies_to_checkmate:
        return "short"
    if len(pv) > plies_to_checkmate:
        return "long"
    board = chess.Board(fen)
    losing_side = 1 if mate > 0 else 0
    try:
        for ply, move in enumerate(pv):
            if ply % 2 == losing_side and board.can_claim_draw():
                return "draw"
            board.push(chess.Move.from_uci(move))
        if board.is_checkmate():
            return "ok"
    except Exception as ex:
        return f"error {ex}"
    return "wrong"


class Analyser:
    def __init__(self, args):
        self.engine = chess.engine.SimpleEngine.popen_uci(args.engine)
        if args.hash is not None:
            self.engine.configure({"Hash": args.hash})
        if args.threads is not None:
            self.engine.configure({"Threads": args.threads})
        if args.syzygyPath is not None:
            self.engine.configure({"SyzygyPath": args.syzygyPath})
        self.limit = chess.engine.Limit(
            nodes=args.nodes, depth=args.depth, time=args.time, mate=args.mate
        )
        self.depthMin = args.depthMin
        self.depthMax = args.depthMax
        self.nodesFill = args.nodesFill
        self.timeFill = args.timeFill
        self.trust = args.trust

    def quit(self):
        self.engine.quit()

    def analyze_fen(self, fen, bm, pv):
        board = chess.Board(fen)
        # first clear hash with a simple d1 search
        self.engine.analyse(board, chess.engine.Limit(depth=1), game=board)
        # now walk to PV leaf node
        ply, pvmate = 0, bm
        for move in pv:
            board.push(chess.Move.from_uci(move))
            ply += 1
            pvmate = -pvmate + (1 if pvmate > 0 else 0)

        # now do a backward analysis, filling the hash table
        max_ply = ply
        while board.move_stack:
            if bool(board.legal_moves):
                depth = min(args.depthMax, max_ply - ply + args.depthMin)
                info = self.engine.analyse(
                    board,
                    chess.engine.Limit(
                        depth=depth, nodes=self.nodesFill, time=self.timeFill
                    ),
                    game=board,
                )
                if "score" in info:
                    score = info["score"].pov(board.turn)
                    depth = info["depth"] if "depth" in info else None
                    nodes = info["nodes"] if "nodes" in info else None
                    print(
                        f"ply {ply:3d}, score {score} (d{depth}, nodes {nodes})",
                        flush=True,
                    )
                    if self.trust:
                        m = score.mate()
                        # we play this safe and only use positive mate scores
                        if m is not None and m > 0 and m <= pvmate and "pv" in info:
                            print(f"Found terminal mate {m}, ending search early.")
                            return bm + m - pvmate, pv[:ply] + [
                                m.uci() for m in info["pv"]
                            ]

            board.pop()
            ply -= 1
            pvmate = -pvmate + (1 if pvmate <= 0 else 0)

        # finally do the actual analysis, to try to prove the mate
        info = self.engine.analyse(board, self.limit, game=board)
        m, pv = None, None
        if "score" in info:
            score = info["score"].pov(board.turn)
            m = score.mate()
            depth = info["depth"] if "depth" in info else None
            nodes = info["nodes"] if "nodes" in info else None
            print(f"Final score {score}, mate {m} (d{depth}, nodes {nodes})")
        if m is not None and abs(m) <= abs(bm) and "pv" in info:
            pv = [m.uci() for m in info["pv"]]

        return m, pv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use conjectured mate PVs from e.g. cdb matetracker to guide local analyses to prove mates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epdFile",
        default="matetrackpv.epd",
        help="file containing positions, their mate scores and possibly proven PVs",
    )
    parser.add_argument(
        "--cdbFile",
        default="../cdbmatetrack/matetrack_cdbpv.epd",
        help="file with conjectured mate PVs",
    )
    parser.add_argument(
        "--trust",
        action="store_true",
        help="take the conjectured PVs as proven (use with care!)",
    )
    parser.add_argument(
        "--outFile",
        default="provenpvs.epd",
        help="output file for newly proven mates with their PVs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output with -v, -vv, -vvv etc.",
    )
    parser.add_argument(
        "--engine",
        default="./stockfish",
        help="name of the engine binary",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        help="nodes limit per position, default: 10**6 without other limits, otherwise None",
    )
    parser.add_argument("--depth", type=int, help="depth limit per puzzle position")
    parser.add_argument("--mate", type=int, help="mate limit per puzzle position")
    parser.add_argument(
        "--time", type=float, help="time limit (in seconds) per puzzle position"
    )
    parser.add_argument("--hash", type=int, help="hash table size in MB")
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() * 3 // 4),
        help="number of threads per position",
    )
    parser.add_argument("--syzygyPath", help="path to syzygy EGTBs")
    parser.add_argument(
        "--depthMin",
        type=int,
        default=15,
        help="search depth increases linearly from PV leaf node to puzzle positions, starting from this value",
    )
    parser.add_argument(
        "--depthMax",
        type=int,
        default=30,
        help="upper cap for search depth for backwards analysis",
    )
    parser.add_argument(
        "--nodesFill",
        type=str,
        help="nodes limit per position for backwards analysis (hash filling)",
    )
    parser.add_argument(
        "--timeFill",
        type=float,
        help="time limit (in seconds) per position for backwards analysis",
    )
    parser.add_argument(
        "--mateType",
        choices=["all", "won", "lost"],
        default="won",
        help="type of positions to find PVs for (WARNING: use all or lost only for reliable engines!)",
    )
    parser.add_argument(
        "--PVstatus",
        default="short+ok",
        help="Filter the PVs to be loaded by status: ok, short, long, draw, wrong, all.",
    )
    args = parser.parse_args()
    if (
        args.nodes is None
        and args.depth is None
        and args.time is None
        and args.mate is None
    ):
        args.nodes = 10**6
    elif args.nodes is not None:
        args.nodes = eval(args.nodes)
    if args.nodesFill is not None:
        args.nodesFill = eval(args.nodesFill)

    p = re.compile("([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")

    d = {}  # prepare "cheat sheet" from cdb mate PVs
    allowed = args.PVstatus.split("+")
    with open(args.cdbFile) as f:
        for line in f:
            m = p.match(line)
            assert m, f"error for line '{line[:-1]}' in file {args.cdbFile}"
            fen, bm = m.group(1), int(m.group(2))
            _, _, pv = line.partition("; PV: ")
            pv, _, _ = pv[:-1].partition(";")  # remove '\n'
            pv = pv.split()
            if (
                args.mateType == "all"
                or args.mateType == "won"
                and bm > 0
                or args.mateType == "lost"
                and bm < 0
            ):
                status = pv_status(fen, bm, pv) if pv else "None"
                if args.verbose:
                    print(f'For "{line[:-1]}" got PV status {status}.')
                if "all" in allowed or status in allowed:
                    d[fen] = bm, pv

    ana_fens = []
    with open(args.epdFile) as f:
        for line in f:
            m = p.match(line)
            if not m:
                print("---------------------> IGNORING : ", line)
            else:
                fen, bm = m.group(1), int(m.group(2))
                _, _, pv = line.partition("; PV: ")
                pv, _, _ = pv[:-1].partition(";")  # remove '\n'
                pv = pv.split()
                if pv_status(fen, bm, pv) != "ok" and fen in d:
                    ana_fens.append((fen, *d[fen], pv))

    total_count = len(ana_fens)
    print(f"Found {total_count} PVs we can use to try to prove/find mate PVs ...")

    ana = Analyser(args)

    count = 0
    with open(args.outFile, "w") as f:
        for i, (fen, bm, pv, oldpv) in enumerate(ana_fens):
            print(f'Analysing {i+1}/{total_count} "{fen}" with bm #{bm}...', flush=True)

            m, pv = ana.analyze_fen(fen, bm, pv)

            if m is None or pv is None:
                continue

            print(f"Found mate #{m}!")
            if abs(m) < abs(bm):
                print(f"Found better mate #{m} for FEN {fen} bm #{bm}")
                bm = m
            else:
                assert m == bm, f"Fatal error: m should be equal to bm but {m} != {bm}"
                if len(pv) <= len(oldpv):
                    print(
                        f"PV has length {len(pv)} <= {len(oldpv)}, so no improvement."
                    )
                    pv = None
            if pv is not None:
                print("Save PV to file.")
                f.write(f"{fen} bm #{bm}; PV: {' '.join(pv)};\n")
                f.close()
                f = open(args.outFile, "a")
                count += 1

    ana.quit()
    print(f"All done. Saved {count} PVs to {args.outFile}.")
