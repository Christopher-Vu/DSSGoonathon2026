"""Address normalization + block-face keys shared by all joins."""
import re

SUFFIX = {
    "AVENUE": "AVE", "AV": "AVE", "STREET": "ST", "BOULEVARD": "BLVD", "DRIVE": "DR", "ROAD": "RD",
    "PLACE": "PL", "COURT": "CT", "TERRACE": "TER", "LANE": "LN", "WAY": "WAY", "WY": "WAY", "ALLEY": "ALY", "AL": "ALY",
    "HIGHWAY": "HWY", "CIRCLE": "CIR", "PLAZA": "PLZ", "SQUARE": "SQ", "PARK": "PARK", "WALK": "WALK",
    "STAIRWAY": "STWY", "EXTENSION": "EXT", "LOOP": "LOOP", "ROW": "ROW", "PARKWAY": "PKWY",
}
ORD = {"FIRST": "01ST", "SECOND": "02ND", "THIRD": "03RD", "FOURTH": "04TH", "FIFTH": "05TH",
       "SIXTH": "06TH", "SEVENTH": "07TH", "EIGHTH": "08TH", "NINTH": "09TH", "TENTH": "10TH"}
_UNIT = re.compile(r"\s+(#|APT|UNIT|STE|SUITE|NO\.?)\s*\S+$")
_NUM = re.compile(r"^(\d+)[A-Z]?(?:\s*-\s*\d+[A-Z]?)?\s+(.*)$")


def norm_street(s: str) -> str:
    """'22nd Avenue' -> '22ND AVE'; '3rd St' -> '03RD ST' (assessor zero-pads 1-digit ordinals)."""
    s = re.sub(r"[.,]", " ", str(s).upper())
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split(" ")
    if toks and toks[-1] in SUFFIX:
        toks[-1] = SUFFIX[toks[-1]]
    toks = [ORD.get(t, t) for t in toks]
    toks = [("0" + t) if re.fullmatch(r"\d(ST|ND|RD|TH)", t) else t for t in toks]
    if len(toks) == 2 and toks[0].isdigit() and toks[1] in ("AVE", "ST"):  # '42 AVE' -> '42ND AVE'
        n = int(toks[0])
        suf = "TH" if 10 <= n % 100 <= 20 else {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")
        toks[0] = f"{n:02d}{suf}"
    return " ".join(toks)


SUFFIXES = set(SUFFIX.values()) | {"HL"}


def bare(street: str) -> str:
    """Street name without its suffix: 'BROADWAY ST' -> 'BROADWAY'."""
    t = street.split(" ")
    return " ".join(t[:-1]) if len(t) > 1 and t[-1] in SUFFIXES else street


def split_addr(a: str):
    """'1523 33rd Avenue #2' -> (1523, '33RD AVE'). Returns (None, None) if no number."""
    if a is None or a != a:
        return None, None
    s = _UNIT.sub("", str(a).upper().strip())
    m = _NUM.match(s)
    if not m:
        return None, None
    return int(m.group(1)), norm_street(m.group(2))


def blockface_from_eviction(a: str):
    """'1200 Block Of 22nd Avenue' -> '22ND AVE|12'. Both sides of the street (no parity in source)."""
    if a is None or a != a:
        return None
    m = re.match(r"^\s*(\d+)\s+BLOCK\s+OF\s+(.*)$", str(a).upper())
    if m:
        return f"{norm_street(m.group(2))}|{int(m.group(1)) // 100}"
    n, st = split_addr(a)
    return None if n is None else f"{st}|{n // 100}"


def blockface(num: int, street: str) -> str:
    return f"{street}|{int(num) // 100}"
