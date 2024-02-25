import argparse, lichess.api, requests, time


def get_lichess_game(fen, db="lichess"):
    url = f"https://explorer.lichess.ovh/{db}?fen={fen}&topGames=1"
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

    if "topGames" in json and json["topGames"] and "id" in json["topGames"][0]:
        return json["topGames"][0]["id"]
    return ""


parser = argparse.ArgumentParser(
    description="Get games from lichess db for FENs in .epd file.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("filename", help="file with FENs")
parser.add_argument("--pgnFile", default="lichess.pgn", help="output file for PGNs")
parser.add_argument(
    "--db",
    choices=["lichess", "master"],
    default="lichess",
    help="lichess db to search in",
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
print(f"Total number of positions: {total}. Looking for games in {args.db} db.")

i = count = 0
with open(args.pgnFile, "w") as f:
    for fen in fens:
        if args.verbose >= 2:
            print(f"FEN = {fen}")
        gameID = get_lichess_game(fen, db=args.db)
        if gameID:
            pgn = lichess.api.game(gameID, format=lichess.format.PGN)
            f.write(pgn)
            count += 1
        if args.verbose:
            i += 1
            print(f"{i}/{total} FENs done. Found {count} games so far.")

print(f"Wrote {count} games that were found to {args.pgnFile}.")
