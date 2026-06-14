import argparse, chess, re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create skeleton _multipv.epd file from an existing .epd file. Output is to stdout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="The source .epd file.")
    args = parser.parse_args()
    p = re.compile(r"^([1-8a-zA-Z/]+ [wb] [a-zA-Z\-]+ [a-h1-8\-]+)( bm #(-?\d+);)?")

    seen = set()
    with open(args.source) as f:
        print("#" * 69)
        print(f"# All the unique positions 1 ply from those in {args.source}.")
        print('# Comment format: "FEN bm #X;" |legalmoves| |moves| |unique|')
        print("#   where legalmoves = FEN's legal moves ...")
        print("#         moves      = those not leading to (check|stale)mate ...")
        print("#         unique     = those leading to yet unseen positions.")
        print("#" * 69)
        for line in f:
            if line.startswith("#"):  # ignore comments
                continue
            m = p.match(line)
            assert m, f"error for line '{line[:-1]}'"
            fen = m.group(1)
            bm = int(m.group(3)) if m.group(2) is not None else None

            children = []
            board = chess.Board(fen)
            lm = mvs = board.legal_moves.count()
            for move in board.legal_moves:
                board.push(move)
                child = board.epd()
                if not bool(board.legal_moves):
                    mvs -= 1
                elif child not in seen:
                    children.append(child)
                board.pop()

            seen.update(children)
            print(f'# FEN "{fen} bm #{bm};" {lm} {mvs} {len(children)}')
            for c in children:
                print(c)
