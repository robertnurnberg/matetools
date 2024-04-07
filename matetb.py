import argparse, chess, collections, time

VALUE_MATE = 30000


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
        print(f"Restrict moves for {'WHITE' if self.mating_side else 'BLACK'} side.")
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
        self.excludePromotionTo = args.excludePromotionTo
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
        if self.excludePromotionTo:
            uci = move.uci()
            if len(uci) == 5 and uci[4] in self.excludePromotionTo:
                return False

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
        print(f"Connect child nodes and score checkmate positions ...")
        dim = len(self.fen2index)
        self.tb = [None] * dim
        for fen, idx in self.fen2index.items():
            board = chess.Board(fen)
            score = -VALUE_MATE if board.is_checkmate() else 0
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
                    if best_score is None or score > best_score:
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
        or args.excludePromotionTo
    ):
        return
    epd = " ".join(args.epd.split()[:4])
    if epd == "8/8/7P/8/pp6/kp6/1p6/1Kb5 w - -":  # bm 7
        args.excludeFrom = "b1"
        args.excludeToCapturable = True
        args.excludeCaptures = True
        args.excludePromotionTo = "qrb"
    elif epd in [
        "8/6Q1/8/7k/8/6p1/6p1/6Kb w - -",  # bm #7
        "8/8/8/8/Q7/5kp1/6p1/6Kb w - -",  # bm #7
    ]:
        args.excludeFrom = "g1"
        args.excludeToCapturable = True
    elif epd == "8/3Q4/8/1r6/kp6/bp6/1p6/1K6 w - -":  # bm #8
        args.excludeFrom = "b1"
        args.excludeTo = "b3"
        args.excludeToCapturable = True
    elif epd == "k7/2Q5/8/2p5/1pp5/1pp5/prp5/nbK5 w - -":  # bm #11
        args.excludeFrom = "c1"
        args.excludeTo = "b2"
        args.excludeToCapturable = True
    elif epd == "8/2P5/8/8/8/1p2k1p1/1p1pppp1/1Kbrqbrn w - -":  # bm #12
        args.firstMove = "c7c8q"
        args.excludeFrom = "b1"
        args.excludeToCapturable = True
    elif epd == "8/8/1p6/1p6/1p6/1p6/pppbK3/rbk3N1 w - -":  # bm #13
        args.excludeFrom = "e2"
        args.excludeToCapturable = True
    elif epd == "8/8/8/2p5/1pp5/brpp4/1pprp2P/qnkbK3 w - -":  # bm #15
        args.excludeFrom = "e1"
        args.excludeToCapturable = True
        args.excludePromotionTo = "qrb"
    elif epd == "4k3/6Q1/8/8/5p2/1p1p1p2/1ppp1p2/nrqrbK2 w - -":  # bm #15
        args.excludeFrom = "f1"
        args.excludeToCapturable = True
    elif epd in [
        "k7/8/1Qp5/2p5/2p5/6p1/2p1ppp1/2Kbrqrn w - -",  # bm #7
        "8/8/8/6r1/8/6B1/p1p5/k1Kb4 w - -",  # bm #16
    ]:
        args.excludeFrom = "c1"
        args.excludeToCapturable = True
    elif epd == "8/8/8/2p5/1pp5/brpp4/qpprp2P/1nkbnK2 w - -":  # bm #16
        args.firstMove = "f1e1"
        args.excludeFrom = "e1"
        args.excludeToCapturable = True
        args.excludePromotionTo = "qrb"
    elif epd == "8/8/8/2p5/1pp5/brpp4/qpprpK1P/1nkbn3 w - -":  # bm #16
        args.firstMove = "f2e1"
        args.excludeFrom = "e1"
        args.excludeToCapturable = True
        args.excludePromotionTo = "qrb"
    elif epd == "8/p7/8/8/8/3p1b2/pp1K1N2/qk6 w - -":  # bm #18
        args.excludeFrom = "d2"
        args.excludeToCapturable = True
    elif epd == "k7/8/1Q6/8/8/6p1/1p1pppp1/1Kbrqbrn w - -":  # bm #26
        args.excludeFrom = "b1"
        args.excludeToCapturable = True
    elif epd in [
        "8/8/2p5/2p5/p1p5/rbp5/p1p2Q2/n1K4k w - -",  # bm #26
        "8/2p5/2p5/8/p1p5/rbp5/p1p2Q2/n1K4k w - -",  # bm #28
    ]:
        args.excludeFrom = "c1"
        args.excludeTo = "a3 c3"
        args.excludeToCapturable = True
    elif epd in [
        "4k3/6Q1/8/5p2/5p2/1p3p2/1ppp1p2/nrqrbK2 w - -",  # bm #17
        "4k3/6Q1/8/8/8/1p3p2/1ppp1p2/nrqrbK2 w - -",  # bm #18
        "8/7p/4k3/5p2/3Q1p2/5p2/5p1p/5Kbr w - -",  # bm #30
    ]:
        args.excludeFrom = "f1"
        args.excludeTo = "h1"
        args.excludeToCapturable = True
    elif epd in [
        "8/8/8/8/6k1/8/2Qp1pp1/3Kbrrb w - -",  # bm #9
        "8/3Q4/8/2kp4/8/1p1p4/pp1p4/rrbK4 w - -",  # bm #12
        "8/8/8/6k1/3Q4/8/3p1pp1/3Kbrrb w - -",  # bm #12
        "k7/8/8/2Q5/3p4/1p1p4/pp1p4/rrbK4 w - -",  # bm #14
        "7k/8/8/8/8/5Qp1/3p1pp1/3Kbrrn w - -",  # bm #16
        "6k1/8/5Q2/8/8/8/3p1pp1/3Kbrrb w - -",  # bm #17
        "4Q3/6k1/8/8/8/8/3p1pp1/3Kbrrb w - -",  # bm #18
        "5k2/8/4Q3/8/8/8/3p1pp1/3Kbrrb w - -",  # bm #18
        "6k1/8/8/8/8/3Q4/3p1pp1/3Kbrrb w - -",  # bm #18
        "8/8/8/1p6/1k6/3Q4/pp1p4/rrbK4 w - -",  # bm #18
        "4k3/8/3Q4/8/8/8/3p1pp1/3Kbrrb w - -",  # bm #19
        "4k3/2Q5/8/8/8/8/3p1pp1/3Kbrrb w - -",  # bm #20
        "8/8/8/8/1Q6/3k4/3p1pp1/3Kbrrb w - -",  # bm #20
        "8/8/6k1/Q7/8/8/3p1pp1/3Kbrrb w - -",  # bm #20
        "8/8/2k5/8/3p4/Qp1p4/pp1p4/rrbK4 w - -",  # bm #20
        "8/3k4/3p1Q2/8/8/1p1p4/pp1p4/rrbK4 w - -",  # bm #23
        "8/1p6/1Q6/8/2kp4/3p4/pp1p4/rrbK4 w - -",  # bm #26
        "8/6p1/4Q3/6k1/8/8/3p1pp1/3Kbrrb w - -",  # bm #29
        "2k5/3p4/1Q6/8/8/1p1p4/pp1p4/rrbK4 w - -",  # bm #30
        "4k3/3p4/5Q2/8/8/1p1p4/pp1p4/rrbK4 w - -",  # bm #30
        "3Q4/8/8/8/k7/8/3p1pp1/3Kbrrb w - -",  # bm #32
        "8/2Q5/8/8/1k1p4/4p1p1/3prpp1/3Kbbrn w - -",  # bm #34
    ]:
        args.excludeFrom = "d1"
        args.excludeToCapturable = True
    elif epd in [
        "8/8/8/1p6/6k1/1p2Q3/p1p1p3/rbrbK3 w - -",  # bm #36
        "8/8/8/1p6/6k1/1Q6/p1p1p3/rbrbK3 b - -",  # bm #-35
    ]:
        args.excludeFrom = "e1"
        args.excludeTo = "a1 c1"
        args.excludeToAttacked = True
    elif epd == "7k/8/5p2/8/8/8/P1Kp1pp1/4brrb w - -":  # bm #43
        args.firstMove = "c2d1"
        args.excludeFrom = "d1"
        args.excludeToAttacked = True
    elif epd == "8/1p6/8/3p3k/3p4/6Q1/pp1p4/rrbK4 w - -":  # bm #46
        args.excludeFrom = "d1"
        args.excludeCaptures = True
        args.excludeToAttacked = True
    elif epd in [
        "6Q1/8/7k/8/8/6p1/4p1pb/4Kbrr w - -",  # bm #12
        "2Q5/k7/8/8/8/8/1pp1p3/brrbK3 w - -",  # bm #16
        "8/8/3p4/1Q6/8/2k5/ppp1p3/brrbK3 w - -",  # bm #22
        "8/1p2k3/8/8/5Q2/8/ppp1p3/qrrbK3 w - -",  # bm #50
        "8/1p2k3/8/8/5Q2/8/ppp1p3/bqrbK3 w - -",  # bm #50
    ]:
        args.excludeFrom = "e1"
        args.excludeToCapturable = True
    elif epd in [
        "8/1p6/4k3/8/3p1Q2/3p4/pp1p4/rrbK4 w - -",  # bm #56
        "8/6pp/5p2/k7/3p4/1Q2p3/3prpp1/3Kbqrb w - -",  # bm #57
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
        "--excludePromotionTo",
        help='String containing piece types that should never be promoted to, e.g. "qrb".',
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
        ("excludePromotionTo", args.excludePromotionTo),
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
