import argparse, lichess.api, requests, time


def get_lichess_game(fen, db="lichess"):
    url = f"https://explorer.lichess.ovh/{db}?fen={fen}"
    if args.verbose >= 3:
        print(f"URL: {url}")
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
                print(f"Take {timeout}s timeout for FEN {fen}")
            time.sleep(timeout)
            timeout *= 2
            assert timeout < 3600, "timeout > 1h, stopping."

    for key in ["topGames", "recentGames"]:
        if key in json and json[key]:
            for g in json[key]:
                if "id" in g:
                    return g["id"]

    if "moves" in json:
        for m in json["moves"]:
            if "game" in m and m["game"] and "id" in m["game"]:
                return m["game"]["id"]

    return ""


parser = argparse.ArgumentParser(
    description="Get games from lichess db for FENs in .epd file (at most one game per FEN).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("filename", help="file with FENs")
parser.add_argument("--pgnFile", default="lichess.pgn", help="output file for PGNs")
parser.add_argument(
    "--db",
    choices=["master", "lichess", "master+lichess"],
    default="master+lichess",
    help="lichess db(s) to search in",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="increase output with -v, -vv, -vvv etc.",
)

args = parser.parse_args()

fens = set()
with open(args.filename) as f:
    for line in f:
        line = line.strip()
        if line:
            if line.startswith("#"):  # ignore comments
                continue
            fen = " ".join(line.split()[:4])
            fens.add(fen)

total = len(fens)
print(
    f"Total number of positions: {total}. Looking for games in {args.db} db.",
    flush=True,
)

dbs = args.db.split("+")
i = count = 0
with open(args.pgnFile, "w", encoding="utf-8") as f:
    for fen in fens:
        if args.verbose >= 2:
            print(f"FEN = {fen}")
        for db in dbs:
            gameID = get_lichess_game(fen, db=db)
            if gameID:
                break
        if gameID:
            pgn = lichess.api.game(gameID, format=lichess.format.PGN)
            f.write(pgn)
            count += 1
        if args.verbose:
            i += 1
            print(f"{i}/{total} FENs done. Found {count} games so far.", flush=True)

print(f"Wrote {count} games that were found to {args.pgnFile}.")
