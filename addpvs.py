import argparse, chess, chess.engine, concurrent.futures, re
from multiprocessing import freeze_support, cpu_count
from time import time
from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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

    def analyze_fens(self, fens):
        result_fens = []
        engine = chess.engine.SimpleEngine.popen_uci(self.engine)
        if self.hash is not None:
            engine.configure({"Hash": self.hash})
        if self.threads is not None:
            engine.configure({"Threads": self.threads})
        if self.syzygyPath is not None:
            engine.configure({"SyzygyPath": self.syzygyPath})
        for fen, bm, pvlength, line in fens:
            pv = None
            board = chess.Board(fen)
            info = engine.analyse(board, self.limit, game=board)
            m = info["score"].pov(board.turn).mate() if "score" in info else None
            if m is not None and abs(m) <= abs(bm) and "pv" in info:
                pv = [m.uci() for m in info["pv"]]
                if m == bm and len(pv) <= pvlength:  # no improvement
                    pv = None
            result_fens.append((fen, bm, m, pv))

        engine.quit()

        return result_fens


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(
        description="Add PVs for mates found by local analysis for positions in e.g. matetrack.epd.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument("--depth", type=int, help="depth limit per position")
    parser.add_argument("--mate", type=int, help="mate limit per position")
    parser.add_argument(
        "--time", type=float, help="time limit (in seconds) per position"
    )
    parser.add_argument("--hash", type=int, help="hash table size in MB")
    parser.add_argument(
        "--threads",
        type=int,
        help="number of threads per position",
    )
    parser.add_argument("--syzygyPath", help="path to syzygy EGTBs")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=cpu_count(),
        help="total number of threads script may use, default: cpu_count()",
    )
    parser.add_argument(
        "--epdFile",
        default="matetrack.epd",
        help="file containing the positions and their mate scores",
    )
    parser.add_argument(
        "--outFile",
        default="matetrackpv.epd",
        help="output file for mates with their mate scores and their PVs",
    )
    parser.add_argument(
        "--mateType",
        choices=["all", "won", "lost"],
        default="won",
        help="type of positions to find PVs for (WARNING: use all or lost only for reliable engines!)",
    )
    args = parser.parse_args()
    if args.nodes is None and args.depth is None and args.time is None:
        args.nodes = 10**6
    elif args.nodes is not None:
        args.nodes = eval(args.nodes)

    ana = Analyser(args)
    p = re.compile("([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")

    print("Loading FENs...")

    fens, ana_fens = [], []
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
                if (
                    args.mateType == "all"
                    or args.mateType == "won"
                    and bm > 0
                    or args.mateType == "lost"
                    and bm < 0
                ) and pv_status(fen, bm, pv) != "ok":
                    ana_fens.append((fen, bm, len(pv), line))
                fens.append((fen, line))

    print(f"{len(fens)} FENs loaded, {len(ana_fens)} need analysis...")

    numfen = len(ana_fens)
    workers = args.concurrency // (args.threads if args.threads else 1)
    assert (
        workers > 0
    ), f"Need concurrency >= threads, but concurrency = {args.concurrency} and threads = {args.threads}."
    fw_ratio = numfen // (4 * workers)
    fenschunked = list(chunks(ana_fens, max(1, fw_ratio)))

    limits = [
        ("nodes", args.nodes),
        ("depth", args.depth),
        ("time", args.time),
        ("mate", args.mate),
        ("hash", args.hash),
        ("threads", args.threads),
        ("syzygyPath", args.syzygyPath),
    ]
    msg = (
        args.engine
        + " on "
        + args.epdFile
        + " with "
        + " ".join([f"--{k} {v}" for k, v in limits if v is not None])
    )

    print(f"\nMate search started for {msg} ...")

    res = []
    futures = []

    with tqdm(total=len(fenschunked), smoothing=0, miniters=1) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as e:
            for entry in fenschunked:
                futures.append(e.submit(ana.analyze_fens, entry))

            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                res += future.result()

    print("")

    d = {}
    count_found = better = 0
    for fen, bm, m, pv in res:
        if m is not None and abs(m) < abs(bm):
            print(f"Found better mate #{m} for FEN {fen} bm #{bm}")
            bm = m
            better += 1
        d[fen] = bm, pv
        count_found += bool(pv is not None)

    print(f"\nUsing {msg}")

    print("Total PVs found:   ", count_found)
    if better:
        print("Total better:   ", better)

    with open(args.outFile, "w") as f:
        for fen, line in fens:
            bm, pv = d.get(fen, (0, None))
            if pv is not None:
                f.write(f"{fen} bm #{bm}; PV: {' '.join(pv)};\n")
            else:
                f.write(line)
