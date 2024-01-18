import argparse, chess, chess.engine, os, re


def pv_status(fen, mate, pv):
    # check if the given pv (list of uci moves) leads to checkmate #mate
    plies_to_checkmate = 2 * mate - 1 if mate > 0 else -2 * mate
    if len(pv) < plies_to_checkmate:
        return "short"
    if len(pv) > plies_to_checkmate:
        return "long"
    board = chess.Board(fen)
    try:
        for move in pv[:-2]:
            board.push(chess.Move.from_uci(move))
        if board.can_claim_draw():
            return "draw"
        for move in pv[-2:]:
            board.push(chess.Move.from_uci(move))
        if board.is_checkmate():
            return "ok"
    except Exception as ex:
        return f"error {ex}"
    return "wrong"


class Analyser:
    def __init__(self, args):
        self.engine = args.engine
        self.limit = chess.engine.Limit(
            nodes=args.nodes, depth=args.depth, time=args.time, mate=args.mate
        )
        self.hash = args.hash
        self.threads = args.threads
        self.syzygyPath = args.syzygyPath
        self.depthMin = args.depthMin
        self.depthMax = args.depthMax

    def analyze_fen(self, fen, bm, pv):
        engine = chess.engine.SimpleEngine.popen_uci(self.engine)
        if self.hash is not None:
            engine.configure({"Hash": self.hash})
        if self.threads is not None:
            engine.configure({"Threads": self.threads})
        if self.syzygyPath is not None:
            engine.configure({"SyzygyPath": self.syzygyPath})
        board = chess.Board(fen)
        # first clear hash with simple d1 search
        engine.analyse(board, chess.engine.Limit(depth=1), game=board)
        # now walk to last but one node
        ply = 0
        for move in pv:
            board.push(chess.Move.from_uci(move))
            ply += 1
        # now do a backward analysis, filling the hash table
        max_ply = ply
        while board.move_stack:
            board.pop()
            ply -= 1
            depth = min(args.depthMax, max_ply - ply + args.depthMin)
            info = engine.analyse(board, chess.engine.Limit(depth=depth), game=board)
            if "score" in info:
                score = info["score"].pov(board.turn)
                print(f"ply {ply:3d}, score {score} (d{depth})")

        # finally do the actual analysis, to try to prove the mate
        info = engine.analyse(board, self.limit, game=board)
        m, pv = None, None
        if "score" in info:
            score = info["score"].pov(board.turn)
            m = score.mate()
            depth = info["depth"] if "depth" in info else None
            print(f"Final score {score}, mate {m} (d{depth})")
        if m is not None and abs(m) <= abs(bm) and "pv" in info:
            pv = [m.uci() for m in info["pv"]]

        engine.quit()

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
        "--outFile",
        default="provenpvs.epd",
        help="output file for newly proven mates with their PVs",
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
            ) and ("all" in allowed or (pv and pv_status(fen, bm, pv) in allowed)):
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
                    print(f"PV has length {len(pv)} < {len(oldpv)}, so no improvement.")
                    pv = None
            if pv is not None:
                print("Save PV to file.")
                f.write(f"{fen} bm #{bm}; PV: {' '.join(pv)};\n")
                f.close()
                f = open(args.outFile, "a")
