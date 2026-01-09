import argparse, random, re, sys, concurrent.futures, chess, chess.engine, chess.syzygy
from time import time
from multiprocessing import freeze_support, cpu_count
from tqdm import tqdm
import json


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Analyser:
    def __init__(self, args):
        self.engine = args.engine
        self.limit = chess.engine.Limit(
            nodes=args.nodes,
            depth=args.depth,
            time=args.time,
        )
        self.limitFill = None
        if (
            (args.nodesFill is None or args.nodesFill)
            and (args.depthFill is None or args.depthFill)
            and (args.timeFill is None or args.timeFill)
        ):
            self.limitFill = chess.engine.Limit(
                nodes=args.nodesFill,
                depth=args.depthFill,
                time=args.timeFill,
            )
        self.hash = args.hash
        self.threads = args.threads
        self.syzygyPath = args.syzygyPath
        self.syzygy50MoveRule = args.syzygy50MoveRule
        self.engineOpts = args.engineOpts

    def analyze_fens(self, fens):
        result_fens = []
        engine = chess.engine.SimpleEngine.popen_uci(self.engine)
        if self.hash is not None:
            engine.configure({"Hash": self.hash})
        if self.threads is not None:
            engine.configure({"Threads": self.threads})
        if self.syzygyPath is not None:
            engine.configure({"SyzygyPath": self.syzygyPath})
        if self.syzygy50MoveRule is not None:
            engine.configure({"Syzygy50MoveRule": self.syzygy50MoveRule})
        if self.engineOpts is not None:
            engine.configure(self.engineOpts)
        for fen, (bm, mating_line) in fens:
            board = chess.Board(fen)
            # first walk the (possibly incomplete) mating line to its leaf node
            for move in mating_line:
                board.push(chess.Move.from_uci(move))

            # now do a retrograde analysis
            while board.move_stack:
                if bool(board.legal_moves) and self.limitFill:
                    engine.analysis(board, self.limitFill, game=board)
                board.pop()

            # finally do the actual analysis for the given fen
            m, pvstr = None, ""
            with engine.analysis(board, self.limit, game=board) as analysis:
                for info in analysis:
                    if "score" in info:
                        if "upperbound" in info or "lowerbound" in info:
                            continue
                        score = info["score"].pov(board.turn)
                        m = score.mate()
                        if m:
                            pv = [m.uci() for m in info["pv"]] if "pv" in info else []
                            pvstr = " ".join(pv)
            result_fens.append((fen, bm, m, pvstr))

        engine.quit()

        return result_fens


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(
        description='Check how many (best) mates an engine finds with retrograde analysis in e.g. matetrackpv.epd, a file with lines of the form "FEN bm #X; PV: m1 m2m3 ...;".',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engine",
        default="./stockfish",
        help="Name of the engine binary.",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        help="Nodes limit per position, default: 10**6 without other limits, otherwise None.",
    )
    parser.add_argument("--depth", type=int, help="depth limit per position")
    parser.add_argument(
        "--time", type=float, help="time limit (in seconds) per position"
    )
    parser.add_argument(
        "--nodesFill",
        type=str,
        help="Nodes limit per position for backward analysis (hash filling), default: 10**5 without other fill limits, otherwise None.",
    )
    parser.add_argument(
        "--depthFill",
        type=str,
        help="Depth limit per position for backward analysis.",
    )
    parser.add_argument(
        "--timeFill",
        type=float,
        help="Time limit (in seconds) per position for backward analysis.",
    )
    parser.add_argument("--hash", type=int, help="hash table size in MB")
    parser.add_argument(
        "--threads",
        type=int,
        help="number of threads per position (values > 1 may lead to non-deterministic results)",
    )
    parser.add_argument(
        "--syzygyPath",
        help="path(s) to syzygy EGTBs, with ':'/';' as separator on Linux/Windows",
    )
    parser.add_argument(
        "--syzygy50MoveRule",
        help='Count cursed wins as wins if set to "False".',
    )
    parser.add_argument(
        "--maxValidMate",
        type=int,
        help="highest possible mate score",
        default=123,  # for SF this is MAX_PLY // 2
    )
    parser.add_argument(
        "--minValidMate",
        type=int,
        help="lowest possible mate score",
        default=-123,  # for SF this is - MAX_PLY // 2
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=cpu_count(),
        help="total number of threads script may use, default: cpu_count()",
    )
    parser.add_argument(
        "--engineOpts",
        type=json.loads,
        help="json encoded dictionary of generic options, e.g. tuning parameters, to be used to initialize the engine",
    )
    parser.add_argument(
        "--epdFile",
        nargs="+",
        default=["matetrackpv.epd"],
        help="file(s) containing the positions, their mate scores and a mating line",
    )
    args = parser.parse_args()
    if args.nodes is None and args.depth is None and args.time is None:
        args.nodes = 10**6
    elif args.nodes is not None:
        args.nodes = eval(args.nodes)
    if args.nodesFill is None and args.depthFill is None and args.timeFill is None:
        args.nodesFill = 10**5
    elif args.nodesFill is not None:
        args.nodesFill = eval(args.nodesFill)
    assert args.syzygy50MoveRule is None or args.syzygy50MoveRule.lower() in [
        "true",
        "false",
    ], "--syzygy50MoveRule expects True/False."

    ana = Analyser(args)
    p = re.compile(r"([0-9a-zA-Z/\- ]*) bm #([0-9\-]*);")

    fens = {}
    for epd in args.epdFile:
        with open(epd) as f:
            for line in f:
                m = p.match(line)
                if not m:
                    print("---------------------> IGNORING : ", line)
                else:
                    fen, bm = m.group(1), int(m.group(2))
                    _, _, pv = line.partition("; PV: ")
                    pv, _, _ = pv[:-1].partition(";")  # remove '\n'
                    pv = pv.split()
                    if fen in fens:
                        bmold, _ = fens[fen]
                        if bm != bmold:
                            print(
                                f'Warning: For duplicate FEN "{fen}" we only keep faster mate between #{bm} and #{bmold}.'
                            )
                            if abs(bm) < abs(bmold):
                                fens[fen] = bm, pv
                    else:
                        fens[fen] = bm, pv

    maxbm = max([abs(bm) for bm, _ in fens.values()]) if fens else 0
    fens = list(fens.items())
    random.seed(42)
    random.shuffle(fens)  # try to balance the analysis time across chunks

    print(f"Loaded {len(fens)} FENs, with max(|bm|) = {maxbm}.")

    numfen = len(fens)
    workers = args.concurrency // (args.threads if args.threads else 1)
    assert (
        workers > 0
    ), f"Need concurrency >= threads, but concurrency = {args.concurrency} and threads = {args.threads}."
    fw_ratio = numfen // (4 * workers)
    fenschunked = list(chunks(fens, max(1, fw_ratio)))

    if args.engineOpts is not None:
        print("Additional generic engine options: ", args.engineOpts)

    limits = [
        ("nodes", args.nodes),
        ("depth", args.depth),
        ("time", args.time),
        ("nodesFill", args.nodesFill),
        ("depthFill", args.depthFill),
        ("timeFill", args.timeFill),
        ("hash", args.hash),
        ("threads", args.threads),
        ("syzygyPath", args.syzygyPath),
        ("syzygy50MoveRule", args.syzygy50MoveRule),
    ]
    msg = (
        args.engine
        + " on "
        + " ".join(args.epdFile)
        + " with "
        + " ".join([f"--{k} {v}" for k, v in limits if v is not None])
    )

    print(f"\nretrocheck started for {msg} ...")
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    name = engine.id.get("name", "")
    engine.quit()

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

    mates = bestmates = tbwins = 0
    issue = {
        "Invalid mate scores": 0,
        "Better mates": 0,
        "Wrong mates": 0,
    }
    for fen, bestmate, mate, pv in res:
        if mate is None:
            continue
        if mate > args.maxValidMate or mate < args.minValidMate:
            issue["Invalid mate scores"] += 1
            print(
                f'Found invalid mate #{mate} outside of [{args.minValidMate}, {args.maxValidMate}] for FEN "{fen}" with bm #{bestmate}.'
            )
        if mate * bestmate > 0:
            mates += 1
            if mate == bestmate:
                bestmates += 1
            if abs(mate) < abs(bestmate):
                issue["Better mates"] += 1
                print(
                    f'Found mate #{mate} (better) for FEN "{fen}" with bm #{bestmate}.'
                )
                print("PV:", pv)
        else:
            issue["Wrong mates"] += 1
            print(
                f'Found mate #{mate} (wrong sign) for FEN "{fen}" with bm #{bestmate}.'
            )
            print("PV:", pv)

    print(f"\nUsing {msg}")
    if name:
        print("Engine ID:    ", name)
    print("Total FENs:   ", numfen)
    print("Found mates:  ", mates)
    print("Best mates:   ", bestmates)
