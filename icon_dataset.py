import numpy as np

# Image dimensions
H = W = 28

# ---------- Utility functions to create shapes ----------

def _grid(h=28, w=28):
    y, x = np.mgrid[0:h, 0:w]
    return y, x

# ---------- Shape functions ----------

def circle(cy, cx, r):
    y, x = _grid(H, W)
    return (x - cx)**2 + (y - cy)**2 <= r**2

def ring(cy, cx, r1, r2):
    y, x = _grid(H, W)
    d2 = (x - cx)**2 + (y - cy)**2
    return (d2 >= r1**2) & (d2 <= r2**2)

def rect(y0, y1, x0, x1):
    y, x = _grid(H, W)
    return (y >= y0) & (y < y1) & (x >= x0) & (x < x1)

def line(y0, x0, y1, x1, thickness=1):
    y, x = _grid(H, W)
    px = x.astype(np.float32); py = y.astype(np.float32)
    x0 = float(x0); y0 = float(y0); x1 = float(x1); y1 = float(y1)
    vx, vy = x1 - x0, y1 - y0
    denom = vx*vx + vy*vy + 1e-8
    t = ((px - x0)*vx + (py - y0)*vy) / denom
    t = np.clip(t, 0.0, 1.0)
    projx = x0 + t*vx
    projy = y0 + t*vy
    d2 = (px - projx)**2 + (py - projy)**2
    return d2 <= (thickness**2)

def triangle(p0, p1, p2):
    # p = (x,y)
    y, x = _grid(H, W)
    x = x.astype(np.float32); y = y.astype(np.float32)
    x0,y0 = p0; x1,y1 = p1; x2,y2 = p2
    den = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2) + 1e-8
    a = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / den
    b = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / den
    c = 1.0 - a - b
    return (a >= 0) & (b >= 0) & (c >= 0)

# -------- utility functions --------

def pm1(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 1, -1).astype(np.int8)

def _dedupe_add(seen: set[bytes], imgs: list[np.ndarray], labs: list[str], img_pm1: np.ndarray, label: str) -> bool:
    key = img_pm1.tobytes()
    if key in seen:
        return False
    seen.add(key)
    imgs.append(img_pm1)
    labs.append(label)
    return True

# -------- icon families with params --------

def make_smiley(rng):
    m = np.zeros((H,W), dtype=bool)
    cx = int(rng.integers(12, 16))
    cy = int(rng.integers(12, 16))
    r = int(rng.integers(8, 11))
    m |= ring(cy, cx, r, r+1)

    eye_dx = int(rng.integers(3, 5))
    eye_y  = cy - int(rng.integers(2, 4))
    m |= circle(eye_y, cx-eye_dx, 1)
    m |= circle(eye_y, cx+eye_dx, 1)

    # smile: two lines (cheap but effective)
    y0 = cy + int(rng.integers(2, 5))
    m |= line(y0, cx-eye_dx, y0+2, cx, thickness=1)
    m |= line(y0+2, cx, y0, cx+eye_dx, thickness=1)
    return pm1(m), "smiley"

def make_sun(rng):
    m = np.zeros((H,W), dtype=bool)
    cx = int(rng.integers(12, 16))
    cy = int(rng.integers(12, 16))
    r = int(rng.integers(4, 6))
    m |= circle(cy, cx, r)
    rays = int(rng.integers(6, 10))
    for k in range(rays):
        ang = 2*np.pi*k/rays + float(rng.uniform(-0.15, 0.15))
        x0 = cx + (r+2)*np.cos(ang); y0 = cy + (r+2)*np.sin(ang)
        x1 = cx + (r+7)*np.cos(ang); y1 = cy + (r+7)*np.sin(ang)
        m |= line(y0, x0, y1, x1, thickness=1)
    return pm1(m), "sun"

def make_house(rng):
    m = np.zeros((H,W), dtype=bool)
    left = int(rng.integers(6, 10))
    right = int(rng.integers(18, 22))
    base_y0 = int(rng.integers(14, 16))
    base_y1 = int(rng.integers(23, 25))
    m |= rect(base_y0, base_y1, left, right)

    roof_peak_y = int(rng.integers(5, 8))
    roof_peak_x = int((left+right)//2 + rng.integers(-1, 2))
    m |= triangle((left, base_y0), (roof_peak_x, roof_peak_y), (right, base_y0))

    # door
    if rng.random() < 0.9:
        dx0 = roof_peak_x - int(rng.integers(1, 3))
        dx1 = dx0 + int(rng.integers(2, 4))
        dy0 = base_y1 - int(rng.integers(5, 7))
        m |= rect(dy0, base_y1, dx0, dx1)
    return pm1(m), "house"

def make_tree(rng):
    m = np.zeros((H,W), dtype=bool)
    trunk_x = int(rng.integers(12, 16))
    trunk_w = int(rng.integers(2, 4))
    m |= rect(17, 26, trunk_x, trunk_x+trunk_w)

    # canopy = 2-4 circles
    n = int(rng.integers(2, 5))
    for _ in range(n):
        cx = int(rng.integers(9, 19))
        cy = int(rng.integers(8, 16))
        r = int(rng.integers(3, 6))
        m |= circle(cy, cx, r)
    return pm1(m), "tree"

def make_rocket(rng):
    m = np.zeros((H,W), dtype=bool)
    cx = int(rng.integers(12, 16))
    top = int(rng.integers(4, 7))
    bottom = int(rng.integers(20, 24))
    body_w = int(rng.integers(4, 6))
    m |= rect(top+2, bottom, cx-body_w, cx+body_w)
    m |= triangle((cx-body_w, top+2), (cx, top), (cx+body_w, top+2))  # nose
    # fins
    m |= triangle((cx-body_w, bottom-3), (cx-body_w-3, bottom), (cx-body_w, bottom))
    m |= triangle((cx+body_w, bottom-3), (cx+body_w, bottom), (cx+body_w+3, bottom))
    # window
    if rng.random() < 0.8:
        m |= circle(int((top+bottom)//2), cx, 2)
    return pm1(m), "rocket"

def make_balloon(rng):
    m = np.zeros((H,W), dtype=bool)
    cx = int(rng.integers(10, 18))
    cy = int(rng.integers(8, 13))
    r = int(rng.integers(4, 7))
    m |= circle(cy, cx, r)
    # knot + string
    m |= triangle((cx-1, cy+r), (cx+1, cy+r), (cx, cy+r+2))
    m |= line(cy+r+2, cx, 26, cx+int(rng.integers(-2,3)), thickness=1)
    return pm1(m), "balloon"

def make_fish(rng):
    m = np.zeros((H,W), dtype=bool)
    cx = int(rng.integers(11, 17))
    cy = int(rng.integers(12, 16))
    rx = int(rng.integers(5, 8))
    ry = int(rng.integers(3, 5))
    y, x = _grid(H, W)
    m |= ((x-cx)**2/(rx*rx) + (y-cy)**2/(ry*ry) <= 1.0)  # ellipse
    # tail
    if rng.random() < 0.5:
        m |= triangle((cx-rx, cy), (cx-rx-5, cy-3), (cx-rx-5, cy+3))
    else:
        m |= triangle((cx+rx, cy), (cx+rx+5, cy-3), (cx+rx+5, cy+3))
    # eye
    m |= circle(cy-1, cx+int(rng.integers(-2,3)), 1)
    return pm1(m), "fish"

def make_umbrella(rng):
    m = np.zeros((H,W), dtype=bool)
    cx = int(rng.integers(12, 16))
    cy = int(rng.integers(9, 12))
    r = int(rng.integers(7, 10))
    # dome (half circle-ish): take circle and cut
    dome = circle(cy, cx, r) & (np.mgrid[0:H,0:W][0] <= cy)
    m |= dome
    # handle
    m |= line(cy, cx, 25, cx, thickness=1)
    m |= line(25, cx, 25, cx+3, thickness=1)
    return pm1(m), "umbrella"

def make_sad_smiley(rng):
    m = np.zeros((H, W), dtype=bool)
    cx = int(rng.integers(12, 16))
    cy = int(rng.integers(12, 16))
    r = int(rng.integers(8, 11))
    m |= ring(cy, cx, r, r + 1)

    eye_dx = int(rng.integers(3, 5))
    eye_y  = cy - int(rng.integers(2, 4))
    m |= circle(eye_y, cx - eye_dx, 1)
    m |= circle(eye_y, cx + eye_dx, 1)

    # sad mouth (inverted V)
    y0 = cy + int(rng.integers(3, 6))
    m |= line(y0 + 2, cx - eye_dx, y0, cx, thickness=1)
    m |= line(y0, cx, y0 + 2, cx + eye_dx, thickness=1)

    # optional tear
    if rng.random() < 0.35:
        m |= circle(eye_y + 2, cx + eye_dx + 1, 1)

    return pm1(m), "sad_smiley"


def make_heart(rng):
    m = np.zeros((H, W), dtype=bool)
    cx = int(rng.integers(12, 16))
    cy = int(rng.integers(10, 14))
    r = int(rng.integers(3, 5))

    # two top lobes
    m |= circle(cy, cx - r, r)
    m |= circle(cy, cx + r, r)

    # bottom triangle
    tip_y = int(rng.integers(20, 24))
    m |= triangle((cx - 2*r - 1, cy + 1), (cx + 2*r + 1, cy + 1), (cx, tip_y))

    # sometimes outline-only heart (thinner)
    if rng.random() < 0.25:
        # crude "outline": remove an inner heart-like blob
        m &= ~((circle(cy, cx - r, max(1, r-1)) | circle(cy, cx + r, max(1, r-1))) |
               triangle((cx - 2*r, cy + 2), (cx + 2*r, cy + 2), (cx, tip_y-2)))

    return pm1(m), "heart"


def make_music_note(rng):
    m = np.zeros((H, W), dtype=bool)

    # stem
    x = int(rng.integers(14, 18))
    y_top = int(rng.integers(6, 10))
    y_bot = int(rng.integers(18, 23))
    m |= rect(y_top, y_bot, x, x + 2)

    # flag (simple triangle/line)
    if rng.random() < 0.5:
        m |= triangle((x + 2, y_top), (x + 9, y_top + 3), (x + 2, y_top + 6))
    else:
        m |= line(y_top, x + 2, y_top + 4, x + 9, thickness=1)

    # note head (filled ellipse-ish via circle stretch trick)
    cx = x - int(rng.integers(4, 7))
    cy = y_bot - int(rng.integers(1, 3))
    r = int(rng.integers(2, 4))
    m |= circle(cy, cx, r) | circle(cy, cx + 1, r)  # slightly wider

    # optional second head (eighth note / beamed)
    if rng.random() < 0.35:
        cx2 = cx + int(rng.integers(6, 9))
        cy2 = cy - int(rng.integers(2, 5))
        m |= circle(cy2, cx2, r) | circle(cy2, cx2 + 1, r)
        # beam
        m |= rect(y_top + 2, y_top + 4, x + 2, x + 10)

    return pm1(m), "note"


def make_catface(rng):
    m = np.zeros((H, W), dtype=bool)

    # fixni stred, nech sa to nerozsype pri okrajoch
    cx = 14
    cy = 15
    r  = int(rng.integers(6, 8))

    # head
    m |= circle(cy, cx, r)

    # ears: two simple filled triangles sitting on top of the head
    ear_h = int(rng.integers(5, 7))
    ear_w = int(rng.integers(4, 6))
    ear_y = cy - r - ear_h + 1  # top of ears

    # left ear
    apex_l = (cx - r + 2, ear_y)                 # (x,y)
    base_l1 = (cx - r - ear_w + 6, ear_y + ear_h)
    base_l2 = (cx - r + ear_w + 2, ear_y + ear_h)
    m |= triangle(base_l1, apex_l, base_l2)

    # right ear
    apex_r = (cx + r - 2, ear_y)
    base_r1 = (cx + r - ear_w - 2, ear_y + ear_h)
    base_r2 = (cx + r + ear_w - 6, ear_y + ear_h)
    m |= triangle(base_r1, apex_r, base_r2)

    # eyes (optional, tiny)
    m |= circle(cy - 2, cx - 3, 1)
    m |= circle(cy - 2, cx + 3, 1)

    # whiskers: three lines each side
    wy = cy + 1
    for dy in (-2, 0, 2):
        m |= line(wy + dy, cx - 2, wy + dy, cx - 10, thickness=1)
        m |= line(wy + dy, cx + 2, wy + dy, cx + 10, thickness=1)

    return pm1(m), "cat"


def make_stick_figure(rng):
    m = np.zeros((H, W), dtype=bool)
    cx = int(rng.integers(12, 16))
    head_y = int(rng.integers(6, 9))
    head_r = int(rng.integers(2, 3))

    # head (ring)
    m |= ring(head_y, cx, head_r, head_r + 1)

    # body
    body_top = head_y + head_r + 2
    body_bot = int(rng.integers(18, 22))
    m |= line(body_top, cx, body_bot, cx, thickness=1)

    # arms
    arm_y = body_top + int(rng.integers(2, 4))
    span = int(rng.integers(6, 9))
    m |= line(arm_y, cx - span, arm_y, cx + span, thickness=1)

    # legs
    leg_y = body_bot
    leg_span = int(rng.integers(4, 7))
    m |= line(leg_y, cx, 26, cx - leg_span, thickness=1)
    m |= line(leg_y, cx, 26, cx + leg_span, thickness=1)

    # optional "pose": one arm up
    if rng.random() < 0.35:
        m |= line(arm_y, cx, arm_y - 5, cx + span, thickness=1)

    return pm1(m), "figure"


def make_chess_piece(rng):
    # simple: pawn / rook / bishop variants
    m = np.zeros((H, W), dtype=bool)
    cx = int(rng.integers(12, 16))
    kind = int(rng.integers(0, 3))

    # base
    m |= rect(22, 25, cx - 8, cx + 8)
    m |= rect(20, 22, cx - 6, cx + 6)

    if kind == 0:  # pawn
        m |= circle(11, cx, 3)
        m |= rect(14, 20, cx - 3, cx + 3)
        lab = "pawn"
    elif kind == 1:  # rook
        m |= rect(9, 20, cx - 4, cx + 4)
        # crenellations
        m |= rect(7, 9, cx - 5, cx + 5)
        m |= rect(6, 7, cx - 5, cx - 3)
        m |= rect(6, 7, cx - 1, cx + 1)
        m |= rect(6, 7, cx + 3, cx + 5)
        lab = "rook"
    else:  # bishop (with notch)
        m |= circle(11, cx, 4)
        m |= rect(14, 20, cx - 3, cx + 3)
        # notch slash
        m &= ~line(8, cx + 2, 15, cx - 2, thickness=1)
        lab = "bishop"

    return pm1(m), f"chess_{lab}"


def make_mug(rng):
    m = np.zeros((H, W), dtype=bool)
    cx = int(rng.integers(11, 15))
    top = int(rng.integers(9, 12))
    bot = int(rng.integers(20, 23))
    left = cx - int(rng.integers(5, 7))
    right = cx + int(rng.integers(5, 7))

    # cup body
    m |= rect(top, bot, left, right)
    # lip
    m |= rect(top-1, top+1, left, right)

    # handle (a ring chunk)
    hx = right + 2
    hy = (top + bot)//2
    m |= ring(hy, hx, 3, 4)
    # open handle by cutting inner part a bit more
    m &= ~circle(hy, hx, 2)

    # steam lines sometimes
    if rng.random() < 0.6:
        for k in range(int(rng.integers(1, 4))):
            x = int(rng.integers(left+1, right-1))
            m |= line(top-2, x, top-7, x + int(rng.integers(-1,2)), thickness=1)

    return pm1(m), "mug"


def make_car(rng):
    m = np.zeros((H, W), dtype=bool)
    cx = int(rng.integers(12, 16))
    y = int(rng.integers(15, 18))
    w = int(rng.integers(8, 10))
    h = int(rng.integers(3, 5))

    # body
    m |= rect(y, y + h, cx - w, cx + w)

    # roof
    roof_h = int(rng.integers(3, 5))
    m |= triangle((cx - w + 2, y), (cx, y - roof_h), (cx + w - 2, y))

    # wheels
    wy = y + h + 2
    wx1 = cx - int(rng.integers(5, 7))
    wx2 = cx + int(rng.integers(5, 7))
    m |= ring(wy, wx1, 1, 2)
    m |= ring(wy, wx2, 1, 2)

    # window cutouts (optional)
    if rng.random() < 0.6:
        m &= ~rect(y, y + 2, cx - 2, cx + 2)

    return pm1(m), "car"

def make_butterfly(rng):
    m = np.zeros((H, W), dtype=bool)
    cx = 14
    cy = int(rng.integers(12, 16))

    # wings: two ellipses (left/right), plus smaller lower wings
    y, x = _grid(H, W)
    x = x.astype(np.float32); y = y.astype(np.float32)

    dx = int(rng.integers(6, 8))
    rx = int(rng.integers(5, 7))
    ry = int(rng.integers(4, 6))

    # upper wings
    left  = ((x-(cx-dx))**2/(rx*rx) + (y-(cy-2))**2/(ry*ry) <= 1.0)
    right = ((x-(cx+dx))**2/(rx*rx) + (y-(cy-2))**2/(ry*ry) <= 1.0)
    m |= left | right

    # lower wings (smaller)
    rx2 = max(3, rx-2)
    ry2 = max(3, ry-2)
    left2  = ((x-(cx-(dx-1)))**2/(rx2*rx2) + (y-(cy+4))**2/(ry2*ry2) <= 1.0)
    right2 = ((x-(cx+(dx-1)))**2/(rx2*rx2) + (y-(cy+4))**2/(ry2*ry2) <= 1.0)
    m |= left2 | right2

    # body
    m |= line(cy-6, cx, cy+9, cx, thickness=1)
    m |= line(cy-6, cx-1, cy+9, cx-1, thickness=1)

    # antennae
    if rng.random() < 0.8:
        m |= line(cy-7, cx, cy-12, cx-4, thickness=1)
        m |= line(cy-7, cx, cy-12, cx+4, thickness=1)

    return pm1(m), "butterfly"

def make_banana(rng):
    m = np.zeros((H, W), dtype=bool)
    y, x = _grid(H, W)
    x = x.astype(np.float32); y = y.astype(np.float32)

    cx = int(rng.integers(13, 15))
    cy = int(rng.integers(13, 15))

    # outer ellipse
    rx = float(rng.integers(10, 12))
    ry = float(rng.integers(6, 8))
    outer = ((x-cx)**2/(rx*rx) + (y-cy)**2/(ry*ry) <= 1.0)

    # inner ellipse shifted to create crescent
    shift = float(rng.integers(3, 5))
    inner = ((x-(cx+shift))**2/((rx-3)*(rx-3)) + (y-(cy-1))**2/((ry-2)*(ry-2)) <= 1.0)

    crescent = outer & (~inner)

    # cut top/bottom a bit to make it more banana-like
    crescent &= (y > 6) & (y < 24)

    # optional small "tips"
    if rng.random() < 0.8:
        crescent |= circle(int(cy-2), int(cx-rx+2), 1)
        crescent |= circle(int(cy+2), int(cx+rx-2), 1)

    m |= crescent
    return pm1(m), "banana"

def make_alien(rng):
    m = np.zeros((H, W), dtype=bool)
    cx = 14
    cy = int(rng.integers(12, 14))

    # head: big oval
    y, x = _grid(H, W)
    x = x.astype(np.float32); y = y.astype(np.float32)

    rx = float(rng.integers(7, 9))
    ry = float(rng.integers(9, 11))
    head = ((x-cx)**2/(rx*rx) + (y-cy)**2/(ry*ry) <= 1.0)
    m |= head

    # eyes: big dark ovals "cut out" (make holes)
    ex = int(rng.integers(4, 6))
    ey = cy - int(rng.integers(1, 3))
    erx = float(rng.integers(2, 3))
    ery = float(rng.integers(3, 4))

    eyeL = ((x-(cx-ex))**2/(erx*erx) + (y-ey)**2/(ery*ery) <= 1.0)
    eyeR = ((x-(cx+ex))**2/(erx*erx) + (y-ey)**2/(ery*ery) <= 1.0)

    # cut eyes out of the head for contrast
    m &= ~(eyeL | eyeR)

    # body: small torso
    body_top = cy + int(ry) - 1
    m |= rect(body_top, body_top + 5, cx - 4, cx + 4)

    # legs
    m |= line(body_top + 4, cx - 2, 27, cx - 4, thickness=1)
    m |= line(body_top + 4, cx + 2, 27, cx + 4, thickness=1)

    # antenna (optional)
    if rng.random() < 0.4:
        m |= line(cy - int(ry) + 1, cx, cy - int(ry) - 4, cx + int(rng.integers(-2,3)), thickness=1)
        m |= circle(cy - int(ry) - 5, cx + int(rng.integers(-2,3)), 1)

    return pm1(m), "alien"


# -------- main dataset generation function --------

FAMILIES = [
    make_smiley, make_sad_smiley,
    make_sun, make_house, make_tree,
    make_rocket, make_balloon, make_fish, make_umbrella,
    make_heart, make_music_note, make_catface, make_stick_figure,
    make_chess_piece, make_mug, make_car, make_butterfly, make_banana, make_alien
]


def generate_icons(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    seen: set[bytes] = set()
    imgs: list[np.ndarray] = []
    labs: list[str] = []

    # generate until we have n unique icons
    while len(imgs) < n:
        fn = FAMILIES[int(rng.integers(0, len(FAMILIES)))]
        img, lab = fn(rng)
        _dedupe_add(seen, imgs, labs, img, lab)

    X = np.stack(imgs, axis=0)  # (n,28,28) int8 in ±1
    y = np.array(labs)
    return X, y