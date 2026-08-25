import argparse, chess, gzip, requests, sys, time


def open_file_rt(filename):
    # allow reading text files either plain or in gzip format
    open_func = gzip.open if filename.endswith(".gz") else open
    return open_func(filename, "rt")


def get_6men_fens(filename):
    fens, withpv = [], 0
    with open_file_rt(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            fen = " ".join(parts[:4])
            count = sum(1 for char in parts[0] if char.lower() in "pnbrqk")
            if count > 6 or (len(parts) > 2 and parts[2] != "-"):
                continue
            _, _, pv = line.partition("; PV: ")
            pv, _, _ = pv[:-1].partition(";")  # remove '\n'
            pv = pv.split()
            fens.append((fen, pv))
            withpv += int(bool(pv))
    return fens, withpv


def get_lichess_json(fen):
    url = f"https://tablebase.lichess.ovh/standard?fen={fen}"
    if args.verbose >= 3:
        print(f"URL: {url}", file=sys.stderr)
    timeout = status = 1

    while status != 200:
        try:
            response = requests.get(url)
            status = response.status_code
            json = response.json()
        except Exception:
            status = 1
        if status != 200:
            if args.verbose >= 2:
                print(f"Take {timeout}s timeout for FEN {fen}", file=sys.stderr)
            time.sleep(timeout)
            timeout *= 2
            assert timeout < 3600, "timeout > 1h, stopping."

    return json


def sanitize_pv(fen, pv):
    board = chess.Board(fen)
    if not bool(board.legal_moves):
        cs = "check" if board.is_checkmate() else "stale"
        print(f'FEN "{fen}" is {cs}mate.', file=sys.stderr)
        return ""
    j = get_lichess_json(fen)
    dtm = j.get("dtm")
    if not dtm:
        return ""
    msg = ""
    pv.reverse()
    pvmoves = []
    while j.get("dtm") and (moves := j.get("moves")):
        bestuci = moves[0]["uci"]
        if pv:
            uci = pv.pop()
            for m in moves:
                if uci == m["uci"]:
                    if m["dtm"] == moves[0]["dtm"]:
                        bestuci = uci
                    else:
                        break
        pvmoves.append(bestuci)
        board.push(chess.Move.from_uci(bestuci))
        if board.is_checkmate():
            walked_dtm = len(pvmoves) if len(pvmoves) % 2 else -len(pvmoves)
            if walked_dtm != dtm:
                print(
                    f'FEN "{fen}" has dtm at root: {dtm}, but length of optimal PV: {abs(walked_dtm)}.',
                    file=sys.stderr,
                )
                dtm = walked_dtm
            bm = (dtm + 1) // 2 if dtm % 2 else dtm // 2
            msg = f" bm #{bm}; PV: " + " ".join(pvmoves) + ";"
            break
        if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
            msg = ""
            break
        j = get_lichess_json(board.fen())  # move counters do matter
    return msg


parser = argparse.ArgumentParser(
    description="Check/extract mate PVs using Lichess DTM EGTBs for FENs in .epd file.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("filename", help="file with FENs")
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="increase output with -v, -vv, -vvv etc.",
)
args = parser.parse_args()

fens, withpv = get_6men_fens(args.filename)
if args.verbose:
    print(f"Total number of positions to process: {len(fens)}.", file=sys.stderr)
    if withpv:
        print(f"Of those with (partial) PV to check: {withpv}.", file=sys.stderr)

for fen, pv in fens:
    print(fen, end="", flush=True)
    msg = sanitize_pv(fen, pv)
    print(msg)
