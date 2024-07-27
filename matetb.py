import argparse, chess, chess.engine, collections, time

VALUE_MATE = 30000


def score2mate(score):
    if score > 0:
        return (VALUE_MATE - score + 1) // 2
    if score < 0:
        return -(VALUE_MATE + score) // 2
    return None


def mate2score(m):
    if m > 0:
        return VALUE_MATE - 2 * m + 1
    if m < 0:
        return -VALUE_MATE - 2 * m
    return None


def filtered_analysis(engine, board, limit=None, game=None):
    info = {}
    with engine.analysis(board, limit, game=game) as analysis:
        for line in analysis:
            if "score" in line and not ("upperbound" in line or "lowerbound" in line):
                info = line
    return info


class MateTB:
    def __init__(self, args):
        self.fen2index = {}  # maps FENs from game tree to their index idx
        self.tb = []  # tb[idx] = [score, children], children a list of indices
        parts = args.epd.split()
        self.root_pos = " ".join(parts[:4])
        playing_side = chess.BLACK if parts[1] == "b" else chess.WHITE
        self.mating_side = not playing_side if " bm #-" in args.epd else playing_side
        print(f"Restrict moves for {'WHITE' if self.mating_side else 'BLACK'} side.")

        self.excludeSANs = [] if args.excludeSANs is None else args.excludeSANs.split()
        self.excludeMoves = (
            [] if args.excludeMoves is None else args.excludeMoves.split()
        )

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

        self.excludeAllowingCapture = args.excludeAllowingCapture
        self.BBexcludeAllowingFrom = self.BBexcludeAllowingTo = 0
        if args.excludeAllowingFrom:
            for square in args.excludeAllowingFrom.split():
                self.BBexcludeAllowingFrom |= chess.BB_SQUARES[
                    chess.parse_square(square)
                ]
        if args.excludeAllowingTo:
            for square in args.excludeAllowingTo.split():
                self.BBexcludeAllowingTo |= chess.BB_SQUARES[chess.parse_square(square)]
        self.excludeAllowingMoves = (
            []
            if args.excludeAllowingMoves is None
            else args.excludeAllowingMoves.split()
        )
        self.excludeAllowingSANs = (
            [] if args.excludeAllowingSANs is None else args.excludeAllowingSANs.split()
        )
        self.needToGenerateResponses = (
            args.excludeToCapturable
            or args.excludeAllowingCapture
            or self.BBexcludeAllowingFrom
            or self.BBexcludeAllowingTo
            or self.excludeAllowingMoves
            or self.excludeAllowingSANs
        )

        self.engine = args.engine
        if self.engine:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine)
            if args.hash:
                self.engine.configure({"Hash": args.hash})
            n = None if args.limitNodes is None else int(args.limitNodes)
            d = None if args.limitDepth is None else int(args.limitDepth)
            t = None if args.limitTime is None else float(args.limitTime)
            self.limit = chess.engine.Limit(nodes=n, depth=d, time=t)
            self.matelimit = None
            if args.mateNodes or args.mateDepth or args.mateTime:
                n = None if args.mateNodes is None else int(args.mateNodes)
                d = None if args.mateDepth is None else int(args.mateDepth)
                t = None if args.mateTime is None else float(args.mateTime)
                self.matelimit = chess.engine.Limit(nodes=n, depth=d, time=t)
            self.analyseAll = args.analyseAll
            self.analyseMoves = (
                [] if args.analyseMoves is None else args.analyseMoves.split()
            )
            self.analyseSANs = (
                [] if args.analyseSANs is None else args.analyseSANs.split()
            )
            self.BBanalyseFrom = self.BBanalyseTo = 0
            if args.analyseFrom:
                for square in args.analyseFrom.split():
                    self.BBanalyseFrom |= chess.BB_SQUARES[chess.parse_square(square)]
            if args.analyseTo:
                for square in args.analyseTo.split():
                    self.BBanalyseTo |= chess.BB_SQUARES[chess.parse_square(square)]

        self.verbose = args.verbose
        self.prepare_opening_book(args.openingMoves)

    def prepare_opening_book(self, openingMoves):
        self.openingBook = {}
        if openingMoves is None:
            return
        print("Preparing the opening book ...")
        lines = []
        for line in openingMoves.split(","):
            stars = line.count("*")
            if stars == 1:
                before, _, after = line.partition("*")
                board = chess.Board(self.root_pos)
                before = before.split()
                for move in before:
                    board.push(chess.Move.from_uci(move))
                length = len(before) + 1
                for move in board.legal_moves:
                    if not before + [str(move)] in [l[:length] for l in lines]:
                        lines.append(before + [str(move)] + after.split())
            elif stars == 0:
                lines.append(line.split())
            else:
                print(f"More than one '*' in line {line}.")
                exit(1)
        for moves in lines:
            if self.verbose >= 3:
                print(f"Processing line {' '.join(moves)} ...")
                if self.verbose >= 4:
                    print(
                        f"https://chessdb.cn/queryc_en/?{self.root_pos} moves {' '.join(moves)}\n".replace(
                            " ", "_"
                        )
                    )

            board = chess.Board(self.root_pos)
            for move in moves:
                if board.turn == self.mating_side:
                    fen = board.epd()
                    if fen in self.openingBook:
                        if self.openingBook[fen] != move:
                            print(
                                f"Cannot specify both {move} and {self.openingBook[fen]} for position {fen}."
                            )
                            exit(1)
                    else:
                        self.openingBook[fen] = move
                m = chess.Move.from_uci(move)
                if m not in board.legal_moves:
                    print(f"Illegal move {m.uci()} in position {fen}.")
                    exit(1)
                board.push(m)
        print(
            f"Done. The opening book contains {len(self.openingBook)} positions/moves."
        )
        if self.verbose >= 4:
            print(f"Opening book: {self.openingBook}.")

    def create_tb(self):
        self.initialize_tb()
        self.connect_children()
        self.generate_tb()

    def allowed_move(self, board, move):
        """restrict the mating side's candidate moves, to reduce the overall tree size"""
        if not board.turn == self.mating_side:
            return True
        uci = move.uci()
        if uci in self.excludeMoves:
            return False
        if board.san(move) in self.excludeSANs:
            return False
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
        if self.excludePromotionTo:
            if len(uci) == 5 and uci[4] in self.excludePromotionTo:
                return False
        if self.needToGenerateResponses:
            board.push(move)
            for m in board.legal_moves:
                if (
                    (
                        self.excludeToCapturable
                        and board.is_capture(m)
                        and m.to_square == move.to_square
                    )
                    or (self.excludeAllowingCapture and board.is_capture(m))
                    or (self.BBexcludeAllowingFrom & (1 << m.from_square))
                    or (self.BBexcludeAllowingTo & (1 << m.to_square))
                    or (m.uci() in self.excludeAllowingMoves)
                    or (board.san(m) in self.excludeAllowingSANs)
                ):
                    board.pop()
                    return False
            board.pop()

        return True

    def analyse_move(self, board, move):
        """decide if to analyse the position resulting from losing side's move"""
        if board.turn == self.mating_side:
            return False
        if self.analyseAll:
            return True
        if move.uci() in self.analyseMoves:
            return True
        if board.san(move) in self.analyseSANs:
            return True
        if self.BBanalyseFrom & (1 << move.from_square):
            return True
        if self.BBanalyseTo & (1 << move.to_square):
            return True
        return False

    def initialize_tb(self):
        tic = time.time()
        print("Create the allowed part of the game tree ...")
        count = 0
        queue = collections.deque([(self.root_pos, False)])
        while queue:
            fen, ana = queue.popleft()
            if fen in self.fen2index:
                continue
            self.fen2index[fen] = count
            count += 1
            if count % 1000 == 0:
                print(f"Progress: {count}", end="\r")
            board = chess.Board(fen)
            score = -VALUE_MATE if board.is_checkmate() else 0
            if score == 0 and self.engine and ana:
                if self.verbose >= 4:
                    print(f'Analysing "{board.epd()}" to {self.limit}.')
                info = filtered_analysis(self.engine, board, self.limit)
                if "score" in info:
                    m = info["score"].pov(board.turn).mate()
                    if m:
                        if self.matelimit:
                            if self.verbose >= 3:
                                print(
                                    f'Found mate {m} analysing "{board.epd()}", doing new analysis to {self.matelimit}.'
                                )
                            info = filtered_analysis(self.engine, board, self.matelimit)
                            if "score" in info:
                                newm = info["score"].pov(board.turn).mate()
                                if newm and abs(newm) < abs(m):
                                    m = newm
                        score = mate2score(m)
                        if self.verbose >= 3:
                            print(
                                f'Found mate {m} analysing "{board.epd()}", setting TB score to {score}.'
                            )
            self.tb.append([score, []])
            if score:
                continue
            onlyMove = self.openingBook.pop(fen, None)
            if self.verbose >= 3 and onlyMove:
                print(f"Picked move {onlyMove} for {fen}.")
                if self.verbose >= 4:
                    print(f"Remaining book: {self.openingBook}.")
            for move in board.legal_moves:
                if onlyMove and move != chess.Move.from_uci(onlyMove):
                    continue
                if onlyMove or self.allowed_move(board, move):
                    analyse = self.engine and self.analyse_move(board, move)
                    board.push(move)
                    queue.append((board.epd(), analyse))
                    board.pop()
        print(f"Found {len(self.fen2index)} positions in {time.time()-tic:.2f}s")

    def connect_children(self):
        tic = time.time()
        print(f"Connect child nodes ...")
        dim = len(self.fen2index)
        for fen, idx in self.fen2index.items():
            if self.tb[idx][0]:  # do not add children to mate nodes
                continue
            board = chess.Board(fen)
            for move in board.legal_moves:
                board.push(move)
                childidx = self.fen2index.get(board.epd(), None)
                if childidx is not None:
                    self.tb[idx][1].append(childidx)
                board.pop()
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

    def write_tb(self, filename):
        with open(filename, "w") as f:
            for fen, idx in self.fen2index.items():
                s = self.tb[idx][0]
                bmstr = "" if s is None else f"bm #{score2mate(s)};"
                f.write(f"{fen} {bmstr}\n")
        print(f"Wrote TB to {filename}.")

    def probe_tb(self, fen):
        idx = self.fen2index.get(fen, None)
        return self.tb[idx][0] if idx is not None else None

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
            score = self.probe_tb(board.epd())
            if score not in [0, None]:
                score = -score + (1 if score > 0 else -1)
            moves.append((score, move))
            board.pop()
        bestscore, bestmove = max(
            moves, key=lambda t: float("-inf") if t[0] is None else t[0]
        )
        score = self.probe_tb(board.epd())
        if bestscore != score:
            # even though we do not assign children to engine mate nodes,
            # transpositions can result in children being present in TB, which
            # would then be found by walking along the PV beyond the mate node
            assert self.engine, "This should never happen."
            if self.verbose >= 4:
                print(
                    f"""Engine mate node "{board.epd()}" has score {score} and children's best score {bestscore}."""
                )
            return ["; PV is short"]
        board.push(bestmove)
        return [str(bestmove)] + self.obtain_pv(board)

    def lengthen_pv(self, pv):
        pv = pv.split()
        board = chess.Board(self.root_pos)
        for move in pv:
            board.push(chess.Move.from_uci(move))
        score = self.probe_tb(board.epd())
        assert score and score > 0, f'Unexpected score {score} for "{board.epd()}".'
        newpv, it = [], 0
        while len(newpv) != len(pv) + VALUE_MATE - score:
            limit = chess.engine.Limit(depth=it) if it else self.limit
            if self.verbose >= 4:
                print(f'Analysing "{board.epd()}" to {limit}.')
            info = filtered_analysis(self.engine, board, limit)
            if "score" in info:
                m = info["score"].pov(board.turn).mate()
                if m:
                    if self.verbose >= 3:
                        print(f'Found mate {m} analysing "{board.epd()}".')
                    s = mate2score(m)
                    if s > score:
                        print("Found shorter mate at end of PV, cannot complete PV.")
                        return ""
                    if s == score and "pv" in info:
                        newpv = pv + [m.uci() for m in info["pv"]]
                        if self.verbose >= 4:
                            print(f"New PV {newpv} has length {len(newpv)}.")
            it += 1
        return " ".join(newpv)

    def output(self):
        board = chess.Board(self.root_pos)
        sp = []
        for move in board.legal_moves:
            board.push(move)
            score = self.probe_tb(board.epd())
            if score not in [0, None]:
                score = -score + (1 if score > 0 else -1)
            pv = [str(move)] + (
                self.obtain_pv(board.copy()) if score not in [0, None] else []
            )
            sp.append((score, pv))
            board.pop()
        sp.sort(reverse=True, key=lambda t: float("-inf") if t[0] is None else t[0])
        score, pv = sp[0][0], " ".join(sp[0][1])
        if score not in [0, None]:
            print("\nMatetrack:")
            print(f"{self.root_pos} bm #{score2mate(score)}; PV: {pv};")
            if self.engine and pv[-14:] == " ; PV is short":
                print("\nLengthening PV ... ", flush=True)
                pv = self.lengthen_pv(pv[:-14])
                if pv:
                    print("\nMatetrack with complete PV:")
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
            pvstr = " ".join(pv)
            if score:
                score_str += f" mate {score2mate(score)}"
                if self.engine and pvstr[-14:] == " ; PV is short":
                    pv = self.lengthen_pv(pvstr[:-14])
                    pvstr = pv if pv else pvstr[:-14]
            elif pv[-1][0] == ";":
                pvstr = " ".join(pv[:-1])
            print(f"multipv {count+1} score {score_str} pv {pvstr}")
            if self.verbose >= 2:
                print(
                    f"https://chessdb.cn/queryc_en/?{self.root_pos} moves {pvstr}\n".replace(
                        " ", "_"
                    )
                )

    def quit(self):
        if self.engine:
            self.engine.quit()


def fill_exclude_options(args):
    """For some known EPDs, this defines the right exclude commands."""
    if (
        args.openingMoves
        or args.excludeMoves
        or args.excludeSANs
        or args.excludeFrom
        or args.excludeTo
        or args.excludeCaptures
        or args.excludeToAttacked
        or args.excludeToCapturable
        or args.excludePromotionTo
        or args.excludeAllowingCapture
        or args.excludeAllowingFrom
        or args.excludeAllowingTo
        or args.excludeAllowingMoves
        or args.excludeAllowingSANs
    ):
        return
    epd = " ".join(args.epd.split()[:4])
    if epd == "8/8/7P/8/pp6/kp6/1p6/1Kb5 w - -":  # bm 7
        args.excludeFrom = "b1"
        args.excludeCaptures = True
        args.excludePromotionTo = "qrb"
        args.excludeToCapturable = True
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
        args.openingMoves = "c7c8q"
        args.excludeFrom = "b1"
        args.excludeToCapturable = True
    elif epd == "8/8/1p6/1p6/1p6/1p6/pppbK3/rbk3N1 w - -":  # bm #13
        args.excludeFrom = "e2"
        args.excludeToCapturable = True
    elif epd == "8/8/8/2p5/1pp5/brpp4/1pprp2P/qnkbK3 w - -":  # bm #15
        args.excludeFrom = "e1"
        args.excludePromotionTo = "qrb"
        args.excludeToCapturable = True
    elif epd == "4k3/6Q1/8/8/5p2/1p1p1p2/1ppp1p2/nrqrbK2 w - -":  # bm #15
        args.excludeFrom = "f1"
        args.excludeToCapturable = True
    elif epd in [
        "8/8/8/6r1/8/6B1/p1p5/k1Kb4 w - -",  # bm #7
        "k7/8/1Qp5/2p5/2p5/6p1/2p1ppp1/2Kbrqrn w - -",  # bm #15
    ]:
        args.excludeFrom = "c1"
        args.excludeToCapturable = True
    elif epd == "8/8/8/2p5/1pp5/brpp4/qpprp2P/1nkbnK2 w - -":  # bm #16
        args.openingMoves = "f1e1"
        args.excludeFrom = "e1"
        args.excludePromotionTo = "qrb"
        args.excludeToCapturable = True
    elif epd == "8/8/8/2p5/1pp5/brpp4/qpprpK1P/1nkbn3 w - -":  # bm #16
        args.openingMoves = "f2e1"
        args.excludeFrom = "e1"
        args.excludePromotionTo = "qrb"
        args.excludeToCapturable = True
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
        args.excludeAllowingCapture = True
    elif epd in [
        "8/8/8/1p6/6k1/1p2Q3/p1p1p3/rbrbK3 w - -",  # bm #36
        "8/8/8/1p6/6k1/1Q6/p1p1p3/rbrbK3 b - -",  # bm #-35
    ]:
        args.excludeFrom = "e1"
        args.excludeTo = "a1 c1"
        args.excludeToAttacked = True
    elif epd == "7k/8/5p2/8/8/8/P1Kp1pp1/4brrb w - -":  # bm #43
        args.openingMoves = "c2d1"
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
        args.excludeAllowingCapture = True
    elif epd in [
        "8/1p6/4k3/8/3p1Q2/3p4/pp1p4/rrbK4 w - -",  # bm #56
        "8/6pp/5p2/k7/3p4/1Q2p3/3prpp1/3Kbqrb w - -",  # bm #57
    ]:
        args.excludeFrom = "d1"
        args.excludeToAttacked = True
    elif epd == "8/8/7p/5K1k/R7/8/8/8 w - -":  # bm #6
        args.excludeAllowingCapture = True
        args.excludeAllowingMoves = "h2h1q"
    elif epd == "8/4p2p/8/8/8/8/6p1/2B1K1kb w - -":  # bm #7
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "g1"
        args.excludeAllowingMoves = "e6e5 e5e4"
    elif epd == "8/7p/7p/7p/1p3Q1p/1Kp5/nppr4/qrk5 w - -":  # bm #54
        args.excludeFrom = "b3"
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "b1 h1"
        args.excludeAllowingMoves = "c3c2"
    elif epd == "5Q2/p1p5/p1p5/6rp/7k/6p1/p1p3P1/rbK5 w - -":  # bm #60 (finds #62)
        args.excludeFrom = "c1 g2"
        args.excludeTo = "a1 g3"
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "h5"
    elif epd == "8/1p4Pp/1p6/1p6/1p5p/5r1k/5p1p/5Kbr w - -":  # bm #72
        args.openingMoves = "g7g8q"
        args.excludeFrom = "f1"
        args.excludeTo = "h1"
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "b3 h5 h4"
    elif epd in [
        "8/6Pp/8/8/7p/5r2/4Kpkp/6br w - -",  # bm #19
        "8/1p4Pp/1p6/1p6/1p5p/5r2/4Kpkp/6br w - -",  # bm #77
    ]:
        args.openingMoves = (
            "g7g8q g2h3 e2f1, "
            + "g7g8q f3g3 g8d5 g3f3 d5f3, "
            + "g7g8q f3g3 g8d5 g2h3 d5e6 g3g4 e2f1, "
            + "g7g8q f3g3 g8d5 g2h3 d5e6 h3g2 e6e4 g3f3 e4f3, "
            + "g7g8q f3g3 g8d5 g2h3 d5e6 h3g2 e6e4 g2h3 e2f1"
        )
        args.excludeFrom = "f1"
        args.excludeTo = "h1"
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "b3 h5 h4"
    elif epd in [
        "8/p7/8/p7/b3Q3/K7/p1r5/rk6 w - -",  # bm #10
        "8/p7/8/p7/b3Q3/K6p/p1r5/rk6 w - -",  # bm #22
        "8/p6p/7p/p6p/b3Q2p/K6p/p1r5/rk6 w - -",  # bm #120
    ]:
        args.excludeFrom = "a3"
        args.excludeTo = "a1"
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "a1 h1"
        args.excludeAllowingSANs = "Kb1 Kc2 Kd1 Kd2"
    elif epd in [
        "8/5P2/8/8/8/n7/1pppp2K/br1r1kn1 w - -",  # bm #10
        "8/3p1P2/8/8/8/n7/1pppp2K/br1r1kn1 w - -",  # bm #28
        "8/2pp1P2/8/8/8/n7/1pppp2K/br1r1kn1 w - -",  # bm #48
        "8/pppp1P2/8/8/8/n7/1pppp2K/br1r1kn1 w - -",  # bm #93
    ]:
        args.openingMoves = (
            "f7f8q g1f3 f8f3 f1e1 f3g3 e1f1 g3g1, "
            + "f7f8q f1e1 f8a3 g1f3 a3f3 * f3g3 e1f1 g3g1, "
            + "f7f8q f1e1 f8a3 g1h3 a3h3 e1f2 h3g3 f2f1 g3g1, "
            + "f7f8q f1e1 f8a3 g1h3 a3h3 * h3g3 e1f1 g3g1, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 g1f3 f8f3 f1e1 f3g3 e1f1 g3g1, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 g1f3 h2g3 d1c1 c5f2 e1d1 f2f3 d1e1 f3h1, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 g1f3 h2g3 f3d4 c5d4 e1f1 d4f2, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 g1f3 h2g3 f3d4 c5d4 * d4g1, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 g1f3 h2g3 * c5f2, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 g1h3 h2h3 e1f1 c5f5 f1g1 f5g4 g1f2 g4g3 f2f1 g3g2 f1e1 g2g1, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 g1h3 h2h3 e1f1 c5f5 f1e1 f5g6 e1f2 g6g3 f2f1 g3g2 f1e1 g2g1, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 g1h3 h2h3 e1f1 c5f5 f1e1 f5g6 e1f1 g6g2 f1e1 g2g1, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 g1h3 h2h3 e1f1 c5f5 f1e1 f5g6 * g6g1, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 g1h3 h2h3 * c5g1, "
            + "f7f8q f1e1 f8a3 e1f1 a3f8 f1e1 f8c5 * c5g1, "
            + "f7f8q f1e1 f8a3 e1f2 a3g3, "
            + "f7f8q f1e1 f8a3 d1c1 a3g3, "
            + "f7f8q f1e1 f8a3 b1c1 a3g3, "
            + "f7f8q f1e1 f8a3 * a3g3 e1f1 g3g1"
        )
        args.excludeSANs = "Kh1 Kg1 Kg2 Kg3 Kg4 Kh4"
        args.excludeTo = "b2 c2 d2 e2"
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "b2 c2 d2 e2"
        args.excludeAllowingSANs = "Ke3 Kf3 Kh1 Kg2 Kh2"
    elif epd in [
        "8/8/6p1/6Pb/p3P1k1/P1p1PNnr/2P1PKRp/7B w - -",  # bm #12
        "8/4p3/6p1/6Pb/p3P1k1/P1p1PNnr/2P1PKRp/7B w - -",  # bm #34
        "8/p1p1p3/2p3p1/6Pb/p3P1k1/P1p1PNnr/2P1PKRp/7B w - -",  # bm #100
    ]:
        args.excludeSANs = "Rf2"
        args.excludeFrom = "f3 e4"
        args.excludeAllowingCapture = True
    elif epd in [
        "n7/1P6/8/8/7p/p6K/3rb3/n6k w - -",  # bm #14 (not yet)
        "n7/1Pp5/8/8/7p/p6K/3rb3/n6k w - -",  # bm #16 (not yet)
        "n7/1P6/5p2/5p2/7p/p6K/3rb3/n6k w - -",  # bm #26 (not yet)
        "n7/pPp5/p4p2/5p2/p6p/p6K/3rb3/n6k w - -",  # bm #110 (not yet)
    ]:
        args.openingMoves = "b7a8q"
        args.excludeFrom = "h3"
        args.excludeTo = "h4"
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "a1"
        args.excludeAllowingSANs = (
            "f1=R c1=R"
            + "Bf1 Bd1 Bd3 Bc4 Bb5 Ba6 Bg4 Bh5"
            + "Be2 Bg2 Bh1 Be4 Bd5 Bc6 Bb7 Ba8"
            + "Kf2 Ke1 Ke2"
        )
    elif epd in [
        "4R3/1n1p4/3n4/8/8/p4p2/7p/5K1k w - -",  # bm #20
        "4R3/1n1p1p2/3n4/8/8/p4p2/7p/5K1k w - -",  # bm #32
        "4R3/pn1p1p1p/p2n4/8/8/p4p2/7p/5K1k w - -",  # bm #69
    ]:
        args.openingMoves = (
            "e8e1 d6e4 e1e4 f3f2 f1f2 * e4e1, e8e1 d6e4 e1e4 * e4e1, e8e1 * f1f2"
        )
        args.excludeSANs = (
            "Ra2 Ra3 Ra4 Ra5 Ra6 Ra7 Ra8 "
            + "Rb2 Rb3 Rb4 Rb5 Rb6 Rb7 Rb8 "
            + "Rc2 Rc3 Rc4 Rc5 Rc6 Rc7 Rc8 "
            + "Rd2 Rd3 Rd4 Rd5 Rd6 Rd7 Rd8 "
            + "Re2 Re3 Re4 Re5 Re6 Re7 Re8 "
            + "Rf2 Rf3 Rf4 Rf5 Rf6 Rf7 Rf8 "
            + "Rg2 Rg3 Rg4 Rg5 Rg6 Rg7 Rg8 "
            + "Rh2 Rh3 Rh4 Rh5 Rh6 Rh7 Rh8 "
        )
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "a1 d1 f1 h1"
        if args.engine is not None:
            args.analyseAll = True
            if not (args.limitNodes or args.limitDepth or args.limitTime):
                args.limitDepth = "2"

    elif epd in [
        "8/8/8/8/NK6/1B1N4/2rpn1pp/2bk1brq w - -",  # bm #7
        "8/7p/8/8/NK6/1B1N4/2rpn1pp/2bk1brq w - -",  # bm #27
        "8/5ppp/5p2/8/NK6/1B1N4/2rpn1pp/2bk1brq w - -",  # bm #87
    ]:
        args.excludeSANs = "Nb6 Nb5 Nc4"
        args.excludeFrom = "a4 b3 d3"
        args.excludeAllowingCapture = True
        if args.engine is None:
            print("For this position --engine needs to be specified.")
            exit(1)
        args.analyseAll = True
        if not (args.limitNodes or args.limitDepth or args.limitTime):
            args.limitDepth = "2"
    elif epd in [
        "7K/8/8/8/4n3/pp1N3p/rp2N1br/bR3n1k w - -",  # bm #3
        "7K/8/8/7p/p3n3/1p1N3p/rp2N1br/bR3n1k w - -",  # bm #31
        "7K/3p4/4p3/1p5p/p3n3/1p1N3p/rp2N1br/bR3n1k w - -",  # bm #96
    ]:
        args.excludeFrom = "d3 e2"
        args.excludeAllowingCapture = True
        args.excludeAllowingFrom = "b2 h2 h1"
        args.excludeAllowingSANs = "Be4 Bd5 Bc6 Bb7 Ba8 Bg4 Bh5"
        if args.engine is None:
            print("For this position --engine needs to be specified.")
            exit(1)
        args.analyseAll = True
        if not (args.limitNodes or args.limitDepth or args.limitTime):
            args.limitDepth = "10"
    elif epd in [
        "r1b5/1pKp4/pP1P4/P6B/3pn3/1P1k4/1P6/5N1N w - -",  # bm #4
        "r1b5/1pKp4/pP1P4/P6B/3pn2p/1P1k4/1P6/5N1N w - -",  # bm #26
        "r1b5/1pKp4/pP1P1p1p/P4p1B/3pn2p/1P1k4/1P6/5N1N w - -",  # bm #121
    ]:
        args.openingMoves = "h5d1"
        args.excludeFrom = "d1 f1 h1 b2 b3 a5 b6 d6"
        args.excludeTo = "c8"
        args.excludeAllowingFrom = "d3 d4 a6 b7 c8 d7"
        args.excludeAllowingTo = "d1 f1 h1"
        if args.engine is None:
            print("For this position --engine needs to be specified.")
            exit(1)
        args.analyseFrom = "e4"
        args.analyseTo = "d1 f1 h1 b2 b3 a5 b6 d6"
        if not (args.limitNodes or args.limitDepth or args.limitTime):
            args.limitDepth = "10"
    elif epd in [
        "n1K5/bNp5/1pP5/1k4p1/1N2pnp1/PP2p1p1/4rpP1/5B2 w - -",  # bm #16
        "n1K5/bNp1p3/1pP5/1k4p1/1N3np1/PP2p1p1/4rpP1/5B2 w - -",  # bm #35
        "n1K5/bNp1p1p1/1pP5/1k6/1N3np1/PP2p1p1/4rpP1/5B2 w - -",  # bm #57
        "n1K5/bNp1p1p1/1pP3p1/1k2p3/1N3n2/PP4p1/4rpP1/5B2 w - -",  # bm #101
    ]:
        args.excludeFrom = "a3 b3 b4 b7 c6 g2"
        args.excludeAllowingToCapturable = True
        args.excludeAllowingFrom = "a8 b5 b6 c7 e2 f1 g3 g2 d3"
        args.excludeTo = "a8"
        args.excludeToCapturable = True
        args.excludeMoves = (
            "f1c4 e2c4 e2d1 e2f3 e2g4 e2h5 f1g2 f1h3 d3c2 d3b1 d3e4 d3f5 d3g6 d3h7"
        )
        if args.engine is None:
            print("For this position --engine needs to be specified.")
            exit(1)
        args.analyseFrom = "f4 d3 g2 h3 h5 g6 e6 d5"
        if not (args.limitNodes or args.limitDepth or args.limitTime):
            args.limitDepth = "2"
    elif epd in [
        "8/8/8/3p2p1/p2np1K1/p3N1pp/rb1N2pr/k1n3Rb w - -",  # bm #4
        "8/8/8/3p2p1/p2np1Kp/p3N1p1/rb1N2pr/k1n3Rb w - -",  # bm #35
        "8/4p3/3p4/p5p1/3n2Kp/p3N1p1/rb1N2pr/k1n3Rb w - -",  # bm #102
    ]:
        args.excludeFrom = "d2 e3 g1"
        args.excludeTo = "g3"
        args.excludeAllowingFrom = "a1 a2 d5"
        args.excludeAllowingCapture = True
        if args.engine is None:
            print("For this position --engine needs to be specified.")
            exit(1)
        args.analyseAll = True
        if not (args.limitNodes or args.limitDepth or args.limitTime):
            args.limitDepth = "10"
    elif epd in [
        "n7/b1p1K3/1pP5/1P6/7p/1p4Pn/1P2N1br/3NRn1k w - -",  # bm #6
        "n7/b1p1K3/1pP5/1P6/6pp/1p4Pn/1P2N1br/3NRn1k w - -",  # bm #9
        "n7/b1p1K3/1pP5/1P4p1/6pp/1p4Pn/1P2N1br/3NRn1k w - -",  # bm #92
        "n7/b1p1K3/1pP4p/1P4p1/6p1/1p4Pn/1P2N1br/3NRn1k w - -",  # bm #126
    ]:
        args.excludeFrom = "b2 d1 e1 b5 c6"
        args.excludeTo = "a8 b6 c7 b3"
        args.excludeMoves = "e2g1 e2c1 e2c3 e2d4 e2f4 g3h1 g3h5 g3f5 g3e4 g3f1"
        args.excludeToCapturable = True
        args.excludePromotionTo = "qrbn"
        args.excludeAllowingFrom = "a8 b6 c7 h2 f1"
        if args.engine is None:
            print("For this position --engine needs to be specified.")
            exit(1)
        args.analyseAll = True
        if not (args.limitNodes or args.limitDepth or args.limitTime):
            args.limitDepth = "10"
    elif epd in [
        "2RN1qN1/5P2/3p1P2/3P4/1K6/1p1p1pp1/1p1p1np1/bk1b2Q1 w - -",  # bm #5
        "2RN1qN1/5P2/3p1P2/3P4/8/Kp1p1pp1/1p1p1np1/bk1b2Q1 w - -",  # bm #21
        "3N1qN1/1Kn2P2/3p1Pp1/3P1pp1/R7/1p1p4/1p1p1n2/bk1b2Q1 w - -",  # bm #107
        "3N1qN1/1Kn2P2/1Q1p1Pp1/3P1pp1/1R6/1p1p4/kp1p4/b2b3n w - -",  # bm #109 (not yet)
    ]:
        if epd == "3N1qN1/1Kn2P2/1Q1p1Pp1/3P1pp1/1R6/1p1p4/kp1p4/b2b3n w - -":
            args.openingMoves = "b4a4 * b6g1"
        args.excludeFrom = "d5 e7 g7 e8"
        args.excludeTo = "d6 a1 b2 b3 d1 d2 d3"
        args.excludeSANs = "Qxf2 Qxf3 Qxf4 Qxf5 Qxf6 Qxf7 Qxg8 Qxg2 Qxg3 Qxg4 Qxg5 Qxg6 Qxg7 Qxg8 Qxh1 Qxh1+ Rb1 Rb2 Rb3 Rb4 Rb5 Rb6 Rb7 Rb8 Rd1 Rd2 Rd3 Rd4 Rd5 Rd6 Rd7 Rd8 Re1 Re2 Re3 Re4 Re5 Re6 Re7 Re8 Rf1 Rf2 Rf3 Rf4 Rf5 Rf6 Rf7 Rf8 Rg1 Rg2 Rg3 Rg4 Rg5 Rg6 Rg7 Rg8 Rh1 Rh2 Rh3 Rh4 Rh5 Rh6 Rh7 Rh8"
        args.excludeMoves = (
            "d8e6 d8c6 d8b7 f7h8 f7h6 f7g5 f7e5 f7d6 g8f6 g8e7 h6g4 h6f5 h6f7 f7f8n"
        )
        args.excludeToCapturable = True
        args.excludePromotionTo = "qrb"
        args.excludeAllowingFrom = (
            "c7 a1 b2 b3 d1 d2 d3 g7 h6 f7 g8 e8 d8 e7 h8 c8 b8 a8"
        )
        args.excludeAllowingTo = "f1 g1 f6 d5"
        args.excludeAllowingMoves = "a2a3 c2c3"
        args.excludeAllowingSANs = "Nxf7 Nxf6 Nxf7+ Nxf6+"
        if args.engine is None:
            print("For this position --engine needs to be specified.")
            exit(1)
        # args.analyseAll = True
        args.analyseFrom = "f8 f2 h1"
        args.analyseTo = "d8 g8 f7"
        if not (args.limitNodes or args.limitDepth or args.limitTime):
            args.limitDepth = "24"
            args.limitNodes = "100000"
            args.mateDepth = "32"


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
        "--openingMoves",
        help="Comma separated opening lines in UCI notation that specify the mating side's moves. In each line a single placeholder '*' is allowed for the defending side.",
    )
    parser.add_argument(
        "--excludeMoves",
        help="Space separated UCI moves that are not allowed.",
    )
    parser.add_argument(
        "--excludeSANs",
        help="Space separated SAN moves that are not allowed.",
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
        help="Never move to a square that risks capture (much slower than --excludeToAttacked).",
    )
    parser.add_argument(
        "--excludePromotionTo",
        help='String containing piece types that should never be promoted to, e.g. "qrb".',
    )
    parser.add_argument(
        "--excludeAllowingCapture",
        action="store_true",
        help="Avoid moves that allow a capture somewhere on the board (much slower than --excludeToAttacked).",
    )
    parser.add_argument(
        "--excludeAllowingFrom",
        help="Space separated square names that opponent's pieces should not be allowed to move from in reply to our move.",
    )
    parser.add_argument(
        "--excludeAllowingTo",
        help="Space separated square names that opponent's pieces should not be allowed to move to in reply to our move.",
    )
    parser.add_argument(
        "--excludeAllowingMoves",
        help="Space separated UCI moves that opponent should not be allowed to make in reply to our move.",
    )
    parser.add_argument(
        "--excludeAllowingSANs",
        help="Space separated SAN moves that opponent should not be allowed to make in reply to our move.",
    )
    parser.add_argument(
        "--outFile",
        help="Optional output file for the TB.",
    )
    parser.add_argument(
        "--engine",
        help="Optional name of the engine binary to analyse positions with the mating side to move to cut off parts of the game tree.",
    )
    parser.add_argument("--hash", type=int, help="hash table size in MB")
    parser.add_argument("--limitNodes", help="engine's nodes limit per position")
    parser.add_argument("--limitDepth", help="engine's depth limit per position")
    parser.add_argument(
        "--limitTime", help="engine's time limit (in seconds) per position"
    )
    parser.add_argument("--mateNodes", help="engine's nodes limit per mate found")
    parser.add_argument("--mateDepth", help="engine's depth limt per mate found")
    parser.add_argument(
        "--mateTime", help="engine's time limit (in seconds) per mate found"
    )
    parser.add_argument(
        "--analyseAll",
        action="store_true",
        help="Analyse all the positions (apart from root) where the mating side is to move.",
    )
    parser.add_argument(
        "--analyseSANs",
        help="Space separated SAN moves of the losing side that are to be analysed by the engine.",
    )
    parser.add_argument(
        "--analyseMoves",
        help="Space separated UCI moves of the losing side that are to be analysed by the engine.",
    )
    parser.add_argument(
        "--analyseFrom",
        help="Moves by the losing side starting from here are to be analysed.",
    )
    parser.add_argument(
        "--analyseTo",
        help="Moves by the losing side going here are to be analysed.",
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
        ("openingMoves", args.openingMoves),
        ("excludeMoves", args.excludeMoves),
        ("excludeSANs", args.excludeSANs),
        ("excludeFrom", args.excludeFrom),
        ("excludeTo", args.excludeTo),
        ("excludeCaptures", args.excludeCaptures),
        ("excludeToAttacked", args.excludeToAttacked),
        ("excludeToCapturable", args.excludeToCapturable),
        ("excludePromotionTo", args.excludePromotionTo),
        ("excludeAllowingCapture", args.excludeAllowingCapture),
        ("excludeAllowingFrom", args.excludeAllowingFrom),
        ("excludeAllowingTo", args.excludeAllowingTo),
        ("excludeAllowingMoves", args.excludeAllowingMoves),
        ("excludeAllowingSANs", args.excludeAllowingSANs),
        ("engine", args.engine),
        ("limitNodes", args.limitNodes),
        ("limitDepth", args.limitDepth),
        ("limitTime", args.limitTime),
        ("mateNodes", args.mateNodes),
        ("mateDepth", args.mateDepth),
        ("mateTime", args.mateTime),
        ("analyseAll", args.analyseAll),
        ("analyseMoves", args.analyseMoves),
        ("analyseSANs", args.analyseSANs),
        ("analyseFrom", args.analyseFrom),
        ("analyseTo", args.analyseTo),
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
    print(f"Running with options {options}", flush=True)
    mtb = MateTB(args)
    mtb.create_tb()
    mtb.output()
    if args.outFile:
        mtb.write_tb(args.outFile)
    mtb.quit()
