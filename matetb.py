import argparse, chess, collections, time

VALUE_MATE = 1000  # larger values mean more iterations when no mate can be found


def score2mate(score):
    if score > 0:
        return (VALUE_MATE - score + 1) // 2
    if score < 0:
        return -(VALUE_MATE + score) // 2
    return None


class MateTB:
    def __init__(self, args):
        parts = args.epd.split()
        self.root_pos = " ".join(parts[:4])
        playing_side = chess.BLACK if parts[1] == "b" else chess.WHITE
        self.mating_side = not playing_side if " bm #-" in args.epd else playing_side
        self.firstMove = args.firstMove
        if self.firstMove and self.mating_side != playing_side:
            print("Cannot specify first move for losing side!")
            exit(1)

        self.BBexcludeFrom = self.BBexcludeTo = 0
        if args.excludeFrom:
            for square in args.excludeFrom.split():
                self.BBexcludeFrom |= chess.BB_SQUARES[chess.parse_square(square)]
        if args.excludeTo:
            for square in args.excludeTo.split():
                self.BBexcludeTo |= chess.BB_SQUARES[chess.parse_square(square)]
        self.excludeCaptures = args.excludeCaptures
        self.excludeToAttacked = args.excludeToAttacked
        self.excludeToCapturable = args.excludeToCapturable
        self.verbose = args.verbose

    def create_tb(self):
        self.get_fen_index()
        self.initialize_tb()
        self.generate_tb()

    def allowed_move(self, board, move):
        """restrict the mating side's candidate moves, to reduce the overall tree size"""
        if not board.turn == self.mating_side:
            return True
        if self.BBexcludeFrom & (1 << move.from_square):
            return False
        if self.BBexcludeTo & (1 << move.to_square):
            return False
        if self.excludeCaptures and board.is_capture(move):
            return False
        if self.excludeToAttacked and board.is_attacked_by(
            not board.turn, move.to_square
        ):
            return False
        if self.excludeToCapturable:
            board.push(move)
            for m in board.legal_moves:
                if board.is_capture(m):
                    board.pop()
                    return False
            board.pop()

        return True

    def get_fen_index(self):
        """fen2index maps the unique FENs from the game tree to integer indices"""
        tic = time.time()
        print("Create the allowed part of the game tree ...")
        count, self.fen2index = 0, {}
        queue = collections.deque([self.root_pos])
        while queue:
            fen = queue.popleft()
            if fen in self.fen2index:
                continue
            self.fen2index[fen] = count
            count += 1
            if count % 1000 == 0:
                print(f"Progress: {count}", end="\r")
            board = chess.Board(fen)
            for move in board.legal_moves:
                if (
                    count == 1
                    and self.firstMove
                    and move != chess.Move.from_uci(self.firstMove)
                ):
                    continue
                if self.allowed_move(board, move):
                    board.push(move)
                    queue.append(board.epd())
                    board.pop()
        print(f"Found {len(self.fen2index)} positions in {time.time()-tic:.2f}s")

    def initialize_tb(self):
        """tb is a list that holds for each fen the score and the child indices"""
        tic = time.time()
        print(f"Connect child nodes and score leaf positions ...")
        dim = len(self.fen2index)
        self.tb = [None] * dim
        for fen, idx in self.fen2index.items():
            board = chess.Board(fen)
            score = (
                -VALUE_MATE
                if board.is_checkmate()
                else 0
                if board.is_stalemate() or board.is_insufficient_material()
                else None
            )
            children = []
            for move in board.legal_moves:
                board.push(move)
                childidx = self.fen2index.get(board.epd(), None)
                if childidx is not None:
                    children.append(childidx)
                board.pop()
            self.tb[idx] = [score, children]
            if (idx + 1) % 1000 == 0:
                print(f"Progress: {idx+1}/{dim}", end="\r")
        print(f"Connected {len(self.tb)} positions in {time.time()-tic:.2f}s")

    def generate_tb(self):
        tic = time.time()
        print("Generate tablebase ...")
        iteration, changed = 0, 1
        while changed:
            changed = 0
            for i in reversed(range(len(self.tb))):
                best_score = None
                for child in self.tb[i][1]:
                    score = self.tb[child][0]
                    if score:
                        score = -score + (1 if score > 0 else -1)
                    if score is not None and (best_score is None or score > best_score):
                        best_score = score
                if best_score is not None and self.tb[i][0] != best_score:
                    self.tb[i][0] = best_score
                    changed += 1
            iteration += 1
            print(f"Iteration {iteration}, changed {changed:9d} scores", end="\r")
        print(
            f"Tablebase generated with {iteration} iterations in {time.time()-tic:.2f}s"
        )

    def obtain_pv(self, board):
        if (
            not bool(board.legal_moves)
            or board.is_insufficient_material()
            or board.can_claim_threefold_repetition()
        ):
            return []
        if not board.turn == self.mating_side and board.can_claim_fifty_moves():
            return ["; draw by 50mr"]
        moves = []
        for move in board.legal_moves:
            board.push(move)
            fen = board.epd()
            idx = self.fen2index.get(fen, None)
            score = self.tb[idx][0] if idx is not None else None
            if score not in [0, None]:
                score = -score + (1 if score > 0 else -1)
            moves.append((score, move))
            board.pop()
        _, bestmove = max(moves, key=lambda t: float("-inf") if t[0] is None else t[0])
        board.push(bestmove)
        return [str(bestmove)] + self.obtain_pv(board)

    def output(self):
        board = chess.Board(self.root_pos)
        sp = []
        for move in board.legal_moves:
            board.push(move)
            fen = board.epd()
            idx = self.fen2index.get(fen, None)
            score = self.tb[idx][0] if idx is not None else None
            if score not in [0, None]:
                score = -score + (1 if score > 0 else -1)
            pv = [str(move)] + (
                self.obtain_pv(board.copy()) if score is not None else []
            )
            sp.append((score, pv))
            board.pop()
        sp.sort(reverse=True, key=lambda t: float("-inf") if t[0] is None else t[0])
        score, pv = sp[0][0], " ".join(sp[0][1])
        if score not in [0, None]:
            print("\nMatetrack:")
            print(f"{self.root_pos} bm #{score2mate(score)}; PV: {pv};")
        else:
            print("No mate found.")
        if self.verbose == 0:
            return
        print("\nMultiPV:")
        for count, (score, pv) in enumerate(sp):
            if score is None:
                print(f"multipv {count+1} score None")
                continue
            score_str = f"cp {score}"
            if score:
                score_str += f" mate {score2mate(score)}"
            pvstr = " ".join(pv)
            print(f"multipv {count+1} score {score_str} pv {pvstr}")
            if self.verbose >= 2:
                if pv[-1] == "; draw by 50mr":
                    pvstr = " ".join(pv[:-1])
                print(
                    f"https://chessdb.cn/queryc_en/?{self.root_pos} moves {pvstr}\n".replace(
                        " ", "_"
                    )
                )


def fill_exclude_options(args):
    """For some known EPDs, this defines the right exclude commands."""
    if (
        args.firstMove
        or args.excludeFrom
        or args.excludeTo
        or args.excludeCaptures
        or args.excludeToAttacked
        or args.excludeToCapturable
    ):
        return
    epd = " ".join(args.epd.split()[:4])
    if epd in [
        "8/8/8/1p6/6k1/1p2Q3/p1p1p3/rbrbK3 w - -",  # bm #36 (success)
        "8/8/8/1p6/6k1/1Q6/p1p1p3/rbrbK3 b - -", # bm #-35 (success)
    ]:
        args.excludeFrom = "e1"
        args.excludeTo = "a1 c1"
        args.excludeToAttacked = True
    elif epd == "7k/8/5p2/8/8/8/P1Kp1pp1/4brrb w - -": # bm #43 (success)
        args.firstMove = "c2d1"
        args.excludeFrom = "d1"
        args.excludeToAttacked = True
    elif epd == "8/1p6/8/3p3k/3p4/6Q1/pp1p4/rrbK4 w - -": # bm #46 (success)
        args.excludeFrom = "d1"
        args.excludeCaptures = True
        args.excludeToAttacked = True
    elif epd in [
        "8/1p6/4k3/8/3p1Q2/3p4/pp1p4/rrbK4 w - -", # bm #56 (success)
        "8/6pp/5p2/k7/3p4/1Q2p3/3prpp1/3Kbqrb w - -", # bm #57 (success)
    ]:
        args.excludeFrom = "d1"
        args.excludeToAttacked = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prove (upper bound) for best mate for a given position by constructing a custom tablebase for a (reduced) game tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epd",
        default="8/8/8/1p6/6k1/1p2Q3/p1p1p3/rbrbK3 w - - bm #36;",
        help="EPD for the root position. If bm is not given, it is assumed that the side to move is mating.",
    )
    parser.add_argument(
        "--firstMove",
        help="Specify the first move if the mating side is the side to move.",
    )
    parser.add_argument(
        "--excludeFrom",
        help="Space separated square names that pieces should never move from.",
    )
    parser.add_argument(
        "--excludeTo",
        help="Space separated square names that pieces should never move to.",
    )
    parser.add_argument(
        "--excludeCaptures",
        action="store_true",
        help="Never capture.",
    )
    parser.add_argument(
        "--excludeToAttacked",
        action="store_true",
        help="Never move to attacked squares (including from pinned pieces, but ignoring en passant).",
    )
    parser.add_argument(
        "--excludeToCapturable",
        action="store_true",
        help="Never move to squares that would allow a capture (much slower than --excludeToAttacked).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output, e.g. -v shows PVs for all legal moves, and -vv also links to chessdb.cn.",
    )
    args = parser.parse_args()
    fill_exclude_options(args)
    options = [
        ("epd", args.epd),
        ("firstMove", args.firstMove),
        ("excludeFrom", args.excludeFrom),
        ("excludeTo", args.excludeTo),
        ("excludeCaptures", args.excludeCaptures),
        ("excludeToAttacked", args.excludeToAttacked),
        ("excludeToCapturable", args.excludeToCapturable),
    ]
    options = " ".join(
        [
            f"--{k}"
            if type(v) == bool
            else f'--{k} "{v}"'
            if " " in v
            else f"--{k} {v}"
            for k, v in options
            if v is not None and str(v) != "False"
        ]
    )
    print(f"Running with options {options}")
    mtb = MateTB(args)
    mtb.create_tb()
    mtb.output()
