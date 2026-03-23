import numpy as np

H = W = 28

# ---------- helpers ----------
def add_black_border(mask01: np.ndarray, border: int = 1) -> np.ndarray:
    m = mask01.copy()
    m[:border, :] = 0
    m[-border:, :] = 0
    m[:, :border] = 0
    m[:, -border:] = 0
    return m

def _center_mask(mask01: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask01)
    if xs.size == 0:
        return mask01.copy()
    cy, cx = ys.mean(), xs.mean()
    ty, tx = (H - 1) / 2.0, (W - 1) / 2.0
    dy, dx = int(round(ty - cy)), int(round(tx - cx))

    out = np.zeros_like(mask01)
    y0s, y1s = max(0, -dy), min(H, H - dy)
    x0s, x1s = max(0, -dx), min(W, W - dx)
    y0d, y1d = max(0, dy),  min(H, H + dy)
    x0d, x1d = max(0, dx),  min(W, W + dx)
    out[y0d:y1d, x0d:x1d] = mask01[y0s:y1s, x0s:x1s]
    return out

def pm1_from_mask(mask01: np.ndarray) -> np.ndarray:
    return np.where(mask01 > 0, 1, -1).astype(np.int8)

# ---------- letters (your 5x7 font) ----------
FONT_5x7 = {
    "A": ["01110","10001","10001","11111","10001","10001","10001"],
    "B": ["11110","10001","10001","11110","10001","10001","11110"],
    "C": ["01111","10000","10000","10000","10000","10000","01111"],
    "D": ["11110","10001","10001","10001","10001","10001","11110"],
    "E": ["11111","10000","10000","11110","10000","10000","11111"],
    "F": ["11111","10000","10000","11110","10000","10000","10000"],
    "G": ["01111","10000","10000","10011","10001","10001","01111"],
    "H": ["10001","10001","10001","11111","10001","10001","10001"],
    "I": ["11111","00100","00100","00100","00100","00100","11111"],
    "J": ["00111","00010","00010","00010","00010","10010","01100"],
    "K": ["10001","10010","10100","11000","10100","10010","10001"],
    "L": ["10000","10000","10000","10000","10000","10000","11111"],
    "M": ["10001","11011","10101","10101","10001","10001","10001"],
    "N": ["10001","11001","10101","10011","10001","10001","10001"],
    "O": ["01110","10001","10001","10001","10001","10001","01110"],
    "P": ["11110","10001","10001","11110","10000","10000","10000"],
    "Q": ["01110","10001","10001","10001","10101","10010","01101"],
    "R": ["11110","10001","10001","11110","10100","10010","10001"],
    "S": ["01111","10000","10000","01110","00001","00001","11110"],
    "T": ["11111","00100","00100","00100","00100","00100","00100"],
    "U": ["10001","10001","10001","10001","10001","10001","01110"],
    "V": ["10001","10001","10001","10001","01010","01010","00100"],
    "W": ["10001","10001","10001","10101","10101","11011","10001"],
    "X": ["10001","01010","00100","00100","00100","01010","10001"],
    "Y": ["10001","01010","00100","00100","00100","00100","00100"],
    "Z": ["11111","00001","00010","00100","01000","10000","11111"],
}

def letter_to_28x28(ch: str, *, scale: int = 4, pad: int = 2, center: bool = True, border: int = 1) -> np.ndarray:
    ch = ch.upper()
    if ch not in FONT_5x7:
        raise ValueError(f"Unknown letter: {ch}")

    glyph = np.array([[c == "1" for c in row] for row in FONT_5x7[ch]], dtype=np.uint8)  # (7,5)
    glyph = np.kron(glyph, np.ones((scale, scale), dtype=np.uint8))  # (28,20) when scale=4

    mask = np.zeros((H, W), dtype=np.uint8)
    gh, gw = glyph.shape
    y0 = (H - gh) // 2
    x0 = min(max(0, pad), W - gw)
    mask[y0:y0+gh, x0:x0+gw] = glyph

    if center:
        mask = _center_mask(mask)
    if border > 0:
        mask = add_black_border(mask, border)

    return pm1_from_mask(mask)

# ---------- digits (7-segment) ----------
SEGMENTS = {
    0: "ABCDEF",
    1: "BC",
    2: "ABGED",
    3: "ABGCD",
    4: "FGBC",
    5: "AFGCD",
    6: "AFGECD",
    7: "ABC",
    8: "ABCDEFG",
    9: "AFGBCD",
}

def digit_to_28x28(d: int, *, thickness: int = 3, margin: int = 4, border: int = 1) -> np.ndarray:
    if d not in SEGMENTS:
        raise ValueError("Digit must be 0–9")

    m = np.zeros((H, W), dtype=np.uint8)
    t = thickness
    mg = margin
    xL, xR = mg, W - mg
    yT, yB = mg, H - mg
    yM = (yT + yB) // 2

    seg = {
        "A": (slice(yT, yT+t), slice(xL+t, xR-t)),
        "D": (slice(yB-t, yB), slice(xL+t, xR-t)),
        "G": (slice(yM-t//2, yM+t//2+1), slice(xL+t, xR-t)),
        "F": (slice(yT+t, yM-t//2), slice(xL, xL+t)),
        "E": (slice(yM+t//2, yB-t), slice(xL, xL+t)),
        "B": (slice(yT+t, yM-t//2), slice(xR-t, xR)),
        "C": (slice(yM+t//2, yB-t), slice(xR-t, xR)),
    }

    for s in SEGMENTS[d]:
        ys, xs = seg[s]
        m[ys, xs] = 1

    if border > 0:
        m = add_black_border(m, border)

    return pm1_from_mask(m)

# ---------- combined dataset ----------
def generate_alnum_dataset(
    *,
    letters: bool = True,
    digits: bool = True,
    border: int = 1,
    letter_scale: int = 4,
    letter_pad: int = 2,
    letter_center: bool = True,
    digit_thickness: int = 3,
    digit_margin: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, 28, 28) int8 in {-1,+1}
      y: (N,) dtype '<U2' labels: 'A'..'Z' and '0'..'9'
    """
    imgs = []
    labs = []

    if letters:
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            imgs.append(letter_to_28x28(ch, scale=letter_scale, pad=letter_pad, center=letter_center, border=border))
            labs.append(ch)

    if digits:
        for d in range(10):
            imgs.append(digit_to_28x28(d, thickness=digit_thickness, margin=digit_margin, border=border))
            labs.append(str(d))

    X = np.stack(imgs, axis=0)
    y = np.array(labs)
    return X, y
