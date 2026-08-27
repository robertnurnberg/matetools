import argparse, chess, gzip, re, requests, sys, time
import chess.chesstb as chesstb

TABLEBASE_URL = "https://huggingface.co/buckets/noobpwnftw/chesstb/resolve/full"
PAGE_SIZE = 512 * 1024


class RemoteSource:
    """Sliceable buffer over HTTP Range requests."""

    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self._size = None
        self._cache = {}

    def _get(self, headers, retries=10, backoff=0.5):
        for attempt in range(retries):
            try:
                resp = self.session.get(self.url, headers=headers, timeout=15)
            except requests.exceptions.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(backoff * 2**attempt)
                continue
            if resp.status_code == 404:
                raise FileNotFoundError(f"{self.url} not found (404)")
            if resp.status_code in (200, 206):
                return resp.content, resp.headers
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(backoff * 2**attempt)
                continue
            raise IOError(f"HTTP {resp.status_code} for {self.url}")

    def _ensure_size(self):
        if self._size is not None:
            return
        data, headers = self._get({"Range": f"bytes=0-{PAGE_SIZE - 1}"})
        cr = headers.get("Content-Range", "")
        self._size = int(cr.split("/")[-1]) if "/" in cr else len(data)
        self._cache[0] = data

    @property
    def size(self):
        self._ensure_size()
        return self._size

    def __len__(self):
        return self.size

    def _page(self, idx):
        if idx not in self._cache:
            start = idx * PAGE_SIZE
            end = min(start + PAGE_SIZE - 1, self.size - 1)
            data, _ = self._get({"Range": f"bytes={start}-{end}"})
            self._cache[idx] = data
        return self._cache[idx]

    def read_range(self, offset, length):
        if offset >= self.size or length <= 0:
            return b""
        end = min(offset + length, self.size)
        start_page = offset // PAGE_SIZE
        end_page = (end - 1) // PAGE_SIZE
        chunks = []
        for p in range(start_page, end_page + 1):
            data = self._page(p)
            p_start = p * PAGE_SIZE
            rel_start = max(0, offset - p_start)
            rel_end = min(len(data), end - p_start)
            chunks.append(data[rel_start:rel_end])
        return b"".join(chunks)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self.size
            if start < 0:
                start = max(0, self.size + start)
            if stop < 0:
                stop = max(0, self.size + stop)
            if stop <= start:
                return b""
            return self.read_range(start, stop - start)
        if isinstance(key, int):
            if key < 0:
                key += self.size
            data = self.read_range(key, 1)
            if not data:
                raise IndexError("Index out of bounds")
            return data[0]
        raise TypeError(f"Invalid key type: {type(key)}")


def patch_remote_seam():
    """Make chess.chesstb read tables from TABLEBASE_URL over HTTP instead
    of a local directory."""
    chesstb._TableFile._open_source = lambda self, path: RemoteSource(path)

    def _custom_find(self, kind, name, ext):
        return f"{TABLEBASE_URL}/{kind}/{name}{ext}"

    chesstb.Tablebase._find = _custom_find

    for open_name, cache_attr in (
        ("_open_wdl", "_wdl_cache"),
        ("_open_dtz", "_dtz_cache"),
        ("_open_dtc", "_dtc_cache"),
        ("_open_dtm", "_dtm_cache"),
        ("_open_dtm50", "_dtm50_cache"),
    ):
        orig_opener = getattr(chesstb.Tablebase, open_name)

        def _wrapped(self, cfg, orig_opener=orig_opener, cache_attr=cache_attr):
            try:
                return orig_opener(self, cfg)
            except FileNotFoundError:
                getattr(self, cache_attr)[cfg.cache_key] = None
                return None

        setattr(chesstb.Tablebase, open_name, _wrapped)


def open_file_rt(filename):
    # allow reading text files either plain or in gzip format
    open_func = gzip.open if filename.endswith(".gz") else open
    return open_func(filename, "rt")


def get_6men_fens_without_cr(filename):
    p = re.compile(r"^([1-8a-zA-Z/]+ [wb] [a-zA-Z\-]+ [a-h1-8\-]+)( bm #(-?\d+);)?")
    fens, withpv = [], 0
    with open_file_rt(filename) as f:
        for line in f:
            if line.startswith("#"):
                continue
            m = p.match(line)
            assert m, f"error for line '{line[:-1]}' in file {filename}"
            fen = m.group(1)
            bm = int(m.group(3)) if m.group(2) is not None else None
            pv = []
            if bm:
                _, _, pv = line.partition("; PV: ")
                pv, _, _ = pv[:-1].partition(";")  # remove '\n'
                pv = pv.split()
            moves = list(reversed(pv))
            root_in_tb = True
            board = chess.Board(fen)
            while chess.popcount(board.occupied) > 6 and moves:
                ucimove = moves.pop()
                board.push(chess.Move.from_uci(ucimove))
                root_in_tb = False
            if (
                chess.popcount(board.occupied) > 6
                or board.has_castling_rights(chess.WHITE)
                or board.has_castling_rights(chess.BLACK)
                or not bool(board.legal_moves)
            ):
                continue
            fens.append((fen, bm, pv, root_in_tb))
            withpv += int(bool(pv))
    return fens, withpv


def get_chesstb_dtm(tb, board):
    # returns None for draws and errors, 0 for checkmate
    if not bool(board.legal_moves):
        return 0 if board.is_checkmate() else None
    try:
        dtm = tb.probe_dtm50(board)
    except chesstb.MissingTableError as exc:
        print(
            f'FEN "{board.fen()}" Error: No tablebase data available: {exc}',
            file=sys.stderr,
        )
        return None
    except (OSError, IOError) as exc:
        print(
            f'FEN "{board.fen()}" Error: Tablebase access error: {exc}', file=sys.stderr
        )
        return None
    return dtm if dtm else None  # chesstb returns 0 for draws


def get_chesstb_child_dtm(tb, board, move, dtm):
    # probes dtm of child position, given dtm value of board
    expected_dtm = -dtm + (1 if dtm > 0 else -1)
    board.push(move)
    child_dtm = get_chesstb_dtm(tb, board)
    board.pop()
    if dtm is not None:
        # tb has a bug if: a move from dtm > 0 leads to too short mated-in,
        # or a move from dtm < 0 leads to draw or too long mate
        if (dtm > 0 and child_dtm is not None and 0 > child_dtm > expected_dtm) or (
            dtm < 0 and (child_dtm is None or child_dtm > expected_dtm or child_dtm < 0)
        ):
            print(
                f'FEN "{board.fen()}" has dtm {dtm}, but {move.uci()} leads to dtm {child_dtm}.',
                file=sys.stderr,
            )
            return None, expected_dtm
    return child_dtm, expected_dtm


def sanitize_pv(tb, fen, bm, pv, root_in_tb):
    board = chess.Board(fen)
    pvmoves = []
    dtm_at_root = None
    if not root_in_tb:
        pv.reverse()
        dtm_at_root = dtm = (2 * bm - 1) if bm > 0 else 2 * bm
        while chess.popcount(board.occupied) > 6 and pv:
            ucimove = pv.pop()
            board.push(chess.Move.from_uci(ucimove))
            pvmoves.append(ucimove)
            dtm = -dtm + (1 if dtm > 0 else -1)

    # now board is a 6men position without cr
    first_dtm = get_chesstb_dtm(tb, board)
    if not first_dtm:
        return ""

    if dtm_at_root is not None:
        if dtm != first_dtm:
            print(
                f'FEN "{board.fen()}" has dtm {first_dtm}, but bm #{bm} suggests dtm {dtm}.',
                file=sys.stderr,
            )
            return f" bm #{bm}; PV: " + " ".join(pvmoves) + ";"
    else:
        pv.reverse()
        dtm_at_root = dtm = first_dtm

    while True:
        bestuci, uci = None, None
        if pv:
            uci = pv.pop()
            child_dtm, expected_dtm = get_chesstb_child_dtm(
                tb, board, chess.Move.from_uci(uci), dtm
            )
            if child_dtm == expected_dtm:
                bestuci = uci
            else:
                pv = []
        if bestuci is None:
            for move in board.legal_moves:
                if move.uci() != uci:
                    child_dtm, expected_dtm = get_chesstb_child_dtm(
                        tb, board, move, dtm
                    )
                if child_dtm == expected_dtm:
                    bestuci = move.uci()
                    break
        if bestuci is None:
            print(
                f'FEN "{board.fen()}" has dtm {dtm}, but could not find move with dtm {expected_dtm}.',
                file=sys.stderr,
            )
            return ""

        pvmoves.append(bestuci)
        board.push(chess.Move.from_uci(bestuci))
        if board.is_checkmate() or child_dtm == 0:
            assert board.is_checkmate() and child_dtm == 0, "Error"
            walked_dtm = len(pvmoves) if len(pvmoves) % 2 else -len(pvmoves)
            if walked_dtm != dtm_at_root:
                print(
                    f'root FEN "{fen}" has dtm {dtm_at_root}, but length of optimal PV: {abs(walked_dtm)}.',
                    file=sys.stderr,
                )
                return ""
            bm = (dtm_at_root + 1) // 2 if dtm_at_root % 2 else dtm_at_root // 2
            return f" bm #{bm}; PV: " + " ".join(pvmoves) + ";"
        if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
            long_fen = fen + " moves " + " ".join(pvmoves)
            print(f'FEN "{long_fen}" is a draw.', file=sys.stderr)
            return ""
        dtm = child_dtm


parser = argparse.ArgumentParser(
    description="Check/extract mate PVs using ChessTB DTM50 EGTBs for FENs in .epd file.",
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
patch_remote_seam()

fens, withpv = get_6men_fens_without_cr(args.filename)
if args.verbose:
    print(f"Total number of positions to process: {len(fens)}.", file=sys.stderr)
    if withpv:
        print(f"Of those with (partial) PV to check: {withpv}.", file=sys.stderr)

with chesstb.open_tablebase(TABLEBASE_URL) as tb:
    for fen, bm, pv, ritb in fens:
        print(fen, end="", flush=True)
        print(sanitize_pv(tb, fen, bm, pv, ritb))
