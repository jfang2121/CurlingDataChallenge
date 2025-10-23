# curling_sim_pygame.py
# Interactive curling end with a dedicated right-side input panel.
# Uses physics from curlingsim.py (Stone, update_stone_motion, etc.)

import math
import pygame
from curlingsim import Stone, update_stone_motion, detect_collision, resolve_collision

# ---------------- WCF sheet geometry (metres) ----------------
L_SHEET = 45.720          # for reference
W_SHEET = 4.750

LINE_W_NARROW = 0.013
LINE_W_HOG    = 0.102

TEE_FROM_MID       = 17.375
BACKLINE_OFF_TEE   = 1.829
HOG_OFF_TEE        = 6.401
COURTESY_OFF_HOG   = 1.219
CENTRE_EXT_TO_HACK = 3.658
WHEELCHAIR_OFF_CL  = 0.457

# House radii (outer -> inner; button RB is the small center circle)
R12 = 1.829
R8  = 1.219
R4  = 0.610
RB  = 0.152

# Stone visuals (physics D=0.29 m in your sim)
STONE_D = 0.29
STONE_R = STONE_D / 2.0
RUNNING_BAND_R = 0.065

# Colors
ICE        = (230, 240, 255)
SHEET_EDGE = (200, 210, 220)
LINE_CLR   = (30, 30, 30)
HOG_CLR    = (220, 60, 60)
WC_CLR     = (100, 100, 140)
HOUSE_BLUE = (60, 100, 200)
HOUSE_RED  = (200, 60, 60)
HOUSE_BTN  = (40, 70, 150)
STONE_BODY = (205, 205, 205)
STONE_BAND = (120, 120, 120)
HANDLE_RED = (220, 50, 50)
HANDLE_YEL = (240, 200, 20)
HUD_TXT    = (255, 255, 255)
BG_OUTSIDE = (10, 25, 45)

MAX_STONES_PER_TEAM = 8
TOTAL_STONES = MAX_STONES_PER_TEAM * 2

# ---- New world frame: x=0 at centerline; y=0 at NEAR tee; +y towards FAR tee.
TEE_TEE = 2 * TEE_FROM_MID   # distance between near and far tees (34.750 m)

# ---------- world helpers (in the new frame) ----------
def tee_positions():
    # y=0 at near tee, y=TEE_TEE at far tee
    return 0.0, TEE_TEE  # near, far

def line_positions():
    tee_near, tee_far = tee_positions()
    near = {
        "tee": tee_near,                         # 0.0
        "backline": tee_near - BACKLINE_OFF_TEE, # negative
        "hog_inner": tee_near + HOG_OFF_TEE,     # positive
        "courtesy": (tee_near + HOG_OFF_TEE) - COURTESY_OFF_HOG,
        "hack": tee_near - CENTRE_EXT_TO_HACK,   # negative (behind tee)
    }
    far = {
        "tee": tee_far,                          # +TEE_TEE
        "backline": tee_far + BACKLINE_OFF_TEE,
        "hog_inner": tee_far - HOG_OFF_TEE,
        "courtesy": (tee_far - HOG_OFF_TEE) + COURTESY_OFF_HOG,
        "hack": tee_far + CENTRE_EXT_TO_HACK,
    }
    return near, far

def compute_view(name):
    """
    World rectangles in the new frame (x=0 centerline, y=0 near tee).
    'full' stays fixed. House views use ±8 m around each tee (like before).
    """
    t_near, t_far = tee_positions()

    # margins so lines/handles aren't cropped in full view
    M_TOP = CENTRE_EXT_TO_HACK + 0.75
    M_BOT = CENTRE_EXT_TO_HACK + 0.75

    HOUSE_HALF = 8.0  # restore the old ~±8 m house zoom

    if name == "full":
        # Full sheet: a little behind near hack to a little past far hack
        return (-W_SHEET/2, -M_BOT, +W_SHEET/2, t_far + M_TOP)

    if name == "house_near":
        return (-W_SHEET/2, t_near - HOUSE_HALF, +W_SHEET/2, t_near + HOUSE_HALF)

    if name == "house_far":
        return (-W_SHEET/2, t_far - HOUSE_HALF, +W_SHEET/2, t_far + HOUSE_HALF)

    if name == "rb_near":
        ymax = R12
        ymin = -3 * R12
        return (-R12, ymin, +R12, ymax)

    if name == "rb_far":
        ymax = t_far + R12
        ymin = t_far - 3 * R12
        return (-R12, ymin, +R12, ymax)

    return (-W_SHEET/2, -M_BOT, +W_SHEET/2, t_far + M_TOP)

# ---------- world <-> screen ----------
def w2s(x, y, scale, ox, oy):
    return int(ox + x * scale), int(oy - y * scale)

def s2w(px, py, scale, ox, oy):
    x = (px - ox) / scale
    y = (oy - py) / scale
    return x, y

def line_w_px(metres, scale):
    return max(1, int(round(metres * scale)))

# ---------- drawing ----------
def draw_house(screen, scale, ox, oy, cx, cy):
    # House rings
    for r_m, col in [(R12, HOUSE_BLUE), (R8, ICE), (R4, HOUSE_RED), (RB, HOUSE_BTN)]:
        pygame.draw.circle(screen, col, w2s(cx, cy, scale, ox, oy), int(round(r_m * scale)))
    # Tee line and button dot
    lw = line_w_px(LINE_W_NARROW, scale)
    x1, y = w2s(-W_SHEET/2, cy, scale, ox, oy)
    x2, _ = w2s(+W_SHEET/2, cy, scale, ox, oy)
    pygame.draw.line(screen, LINE_CLR, (x1, y), (x2, y), lw)
    pygame.draw.circle(screen, LINE_CLR, w2s(0.0, cy, scale, ox, oy), max(2, lw))

def draw_sheet(screen, world_rect, scale, ox, oy):
    xmin, ymin, xmax, ymax = world_rect
    tl = w2s(xmin, ymax, scale, ox, oy)
    br = w2s(xmax, ymin, scale, ox, oy)
    pygame.draw.rect(screen, ICE, (*tl, br[0]-tl[0], br[1]-tl[1]))
    pygame.draw.rect(screen, SHEET_EDGE, (*tl, br[0]-tl[0], br[1]-tl[1]), 1)

def draw_lines(screen, scale, ox, oy, world_rect):
    xmin, ymin, xmax, ymax = world_rect
    near, far = line_positions()
    lw = line_w_px(LINE_W_NARROW, scale)

    # Wheelchair verticals near hog lines
    for end, toward in [(near, +1), (far, -1)]:
        y_hog = end["hog_inner"]
        y_edge = end["tee"] + toward * R12
        for sign in (-1, +1):
            x_w = sign * WHEELCHAIR_OFF_CL
            pygame.draw.line(screen, WC_CLR,
                             w2s(x_w, y_hog, scale, ox, oy),
                             w2s(x_w, y_edge, scale, ox, oy), lw)

    # Hog lines (thick)
    hog_w = line_w_px(LINE_W_HOG, scale)
    for y in (near["hog_inner"], far["hog_inner"]):
        x1, yy = w2s(-W_SHEET/2, y, scale, ox, oy)
        x2, _  = w2s(+W_SHEET/2, y, scale, ox, oy)
        pygame.draw.line(screen, HOG_CLR, (x1, yy), (x2, yy), hog_w)

    # Courtesy lines
    for y in (near["courtesy"], far["courtesy"]):
        pygame.draw.line(screen, LINE_CLR,
                         w2s(-W_SHEET/2, y, scale, ox, oy),
                         w2s(+W_SHEET/2, y, scale, ox, oy), lw)

    # Back lines
    for y in (near["backline"], far["backline"]):
        pygame.draw.line(screen, LINE_CLR,
                         w2s(-W_SHEET/2, y, scale, ox, oy),
                         w2s(+W_SHEET/2, y, scale, ox, oy), lw)

    # Hacks
    hack_len = 0.457
    for y in (near["hack"], far["hack"]):
        pygame.draw.line(screen, LINE_CLR,
                         w2s(-hack_len/2, y, scale, ox, oy),
                         w2s(+hack_len/2, y, scale, ox, oy), lw)

    # Houses
    draw_house(screen, scale, ox, oy, 0.0, near["tee"])  # near house at y=0
    draw_house(screen, scale, ox, oy, 0.0, far["tee"])   # far house at y=TEE_TEE

    # Centerline (through entire current view)
    x = w2s(0.0, 0.0, scale, ox, oy)[0]
    y_top = w2s(0.0, ymax, scale, ox, oy)[1]
    y_bot = w2s(0.0, ymin, scale, ox, oy)[1]
    pygame.draw.line(screen, LINE_CLR, (x, y_bot), (x, y_top), lw)

def draw_stone(screen, scale, ox, oy, s, handle_color):
    px, py = w2s(s.x, s.y, scale, ox, oy)
    rr = max(1, int(round(STONE_R * scale)))
    pygame.draw.circle(screen, STONE_BODY, (px, py), rr)
    band_r = int(round(RUNNING_BAND_R * scale))
    if band_r < rr:
        pygame.draw.circle(screen, STONE_BAND, (px, py), rr, max(1, rr - band_r))
    h_r = int(rr * 0.72)
    phi = getattr(s, "phi", 0.0)
    hx = px + int(h_r * math.cos(phi))
    hy = py - int(h_r * math.sin(phi))
    pygame.draw.line(screen, handle_color, (px, py), (hx, hy), max(2, rr // 8))
    pygame.draw.circle(screen, (50, 50, 50), (px, py), max(2, rr // 10))

# ---------- layout (THREE PANELS) ----------
def layout_three_panels(screen, main_world, panel_world, gap_px=10, pad_px=12, right_w_px=360, mid_w_px=460):
    """
    Returns:
      main_surf, main_rect,
      mid_surf,  mid_rect,
      right_surf,right_rect,
      (m_scale, m_ox, m_oy),
      (mid_scale, mid_ox, mid_oy)
    """
    W, H = screen.get_size()

    # Size the middle (zoomed) and right (UI) columns
    right_rect = pygame.Rect(W - right_w_px, 0, right_w_px, H)
    mid_rect   = pygame.Rect(W - right_w_px - gap_px - mid_w_px, 0, mid_w_px, H)
    main_rect  = pygame.Rect(0, 0, max(100, mid_rect.left - gap_px), H)

    main_surf  = screen.subsurface(main_rect)
    mid_surf   = screen.subsurface(mid_rect)
    right_surf = screen.subsurface(right_rect)

    def scale_origin(world, surf, pad):
        sw, sh = surf.get_size()
        xmin, ymin, xmax, ymax = world
        ww, wh = xmax - xmin, ymax - ymin
        # Fixed mapping inside panel: keep aspect by using height
        scale = (sh - 2*pad) / wh
        ox = pad - xmin * scale
        oy = sh - pad + ymin * scale
        return scale, ox, oy

    return (main_surf, main_rect,
            mid_surf,  mid_rect,
            right_surf,right_rect,
            scale_origin(main_world, main_surf, pad_px),
            scale_origin(panel_world, mid_surf, pad_px))

# ---------- sim helpers ----------
def any_moving(stones, v_eps=1e-3):
    return any(getattr(s, "moving", True) and s.v > v_eps for s in stones)

def aim_at(x0, y0, x1, y1):
    return math.atan2(y1 - y0, x1 - x0)

# ---------- scoring ----------
def in_house(s, tee_y):
    d = math.hypot(s.x - 0.0, s.y - tee_y)
    return d <= (R12 + STONE_R)

def house_score(stones, tee_y):
    tagged = []
    for s in stones:
        if not in_house(s, tee_y):
            continue
        color = "Red" if (getattr(s, "_team", 0) == 0) else "Yellow"
        d = math.hypot(s.x - 0.0, s.y - tee_y)
        tagged.append((d, color))
    if not tagged:
        return None, 0, {"reason": "No stones in house."}
    tagged.sort(key=lambda t: t[0])
    best_color = tagged[0][1]
    opp_color  = "Yellow" if best_color == "Red" else "Red"
    opp_closest = next((d for d, c in tagged if c == opp_color), float("inf"))
    score = sum(1 for d, c in tagged if c == best_color and d < opp_closest)
    return best_color, score, {"opp_closest": opp_closest, "counted": score}

# ---------- simple UI widgets ----------
class InputBox:
    def __init__(self, rect, text="", placeholder="", numeric=True):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.placeholder = placeholder
        self.active = False
        self.numeric = numeric
        self.invalid = False

    def handle_event(self, evt):
        if evt.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(evt.pos)
        elif evt.type == pygame.KEYDOWN and self.active:
            if evt.key == pygame.K_RETURN:
                # Let caller know Enter was pressed by returning a sentinel string
                return "__ENTER__"
            elif evt.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                ch = evt.unicode
                if not ch:
                    return
                if self.numeric:
                    # allow digits, one dot, one leading minus
                    if ch.isdigit() or (ch == '.' and '.' not in self.text) or (ch == '-' and len(self.text) == 0):
                        self.text += ch
                else:
                    self.text += ch
        return None

    def draw(self, surf, font):
        base_color = (240, 240, 245)
        border_color = (80, 120, 200) if self.active else (150, 150, 160)
        if self.invalid:
            border_color = (200, 80, 80)
        pygame.draw.rect(surf, base_color, self.rect, border_radius=8)
        pygame.draw.rect(surf, border_color, self.rect, 2, border_radius=8)
        txt = self.text if self.text else self.placeholder
        col = (15, 20, 30) if self.text else (120, 130, 150)
        surf.blit(font.render(txt, True, col), (self.rect.x + 10, self.rect.y + 6))

    def get_value(self, default=None):
        if self.text == "":
            self.invalid = True
            return default, False
        try:
            v = float(self.text)
            self.invalid = False
            return v, True
        except ValueError:
            self.invalid = True
            return default, False

class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def handle_event(self, evt):
        if evt.type == pygame.MOUSEBUTTONDOWN and evt.button == 1:
            if self.rect.collidepoint(evt.pos):
                return True
        return False

    def draw(self, surf, font):
        base = (235, 240, 250)
        border = (120, 140, 190)
        pygame.draw.rect(surf, base, self.rect, border_radius=12)
        pygame.draw.rect(surf, border, self.rect, 2, border_radius=12)
        label_surf = font.render(self.label, True, (25, 35, 55))
        surf.blit(label_surf, (self.rect.centerx - label_surf.get_width()//2,
                               self.rect.centery - label_surf.get_height()//2))

# ---------- main loop ----------
def run_pygame_rink(stones_init=None, title="Interactive Curling End",
                    width=1400, height=820, fps=60, start_view="full"):
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 18)
    font_big = pygame.font.SysFont(None, 22)

    # Always start with full view on the left; will not zoom while typing
    view_main = "full"

    tee_near, tee_far = tee_positions()
    near, _ = line_positions()

    stones = list(stones_init or [])
    for idx, s in enumerate(stones):
        setattr(s, "_team", idx % 2)  # 0=Red,1=Yellow

    # turn/throw tracking
    next_team = len(stones) % 2
    thrown_red = sum(1 for s in stones if getattr(s, "_team", 0) == 0)
    thrown_yel = sum(1 for s in stones if getattr(s, "_team", 0) == 1)

    # Default release limits (centerline, roughly between hack and near hog)
    y_min_throw = near["hack"] + 0.10
    y_max_throw = near["hog_inner"] - 0.10
    x_release = 0.0
    y_release = (y_min_throw + y_max_throw) / 2.0

    # Right-side input widgets (include X and Y)
    x_box     = InputBox((0, 0, 200, 34), text=f"{x_release:.3f}", placeholder="X (m) e.g. 0.000")
    y_box     = InputBox((0, 0, 200, 34), text=f"{y_release:.3f}", placeholder="Y (m) e.g. 0.500")
    speed_box = InputBox((0, 0, 200, 34), text="", placeholder="Speed (m/s) e.g. 3.0")
    angle_box = InputBox((0, 0, 200, 34), text="", placeholder="Angle ψ (deg) e.g. 90")
    spin_box  = InputBox((0, 0, 200, 34), text="", placeholder="Spin ω (rad/s) e.g. +3.0")
    btn_add   = Button((0, 0, 200, 40), "Add Stone")

    # prevent left panel view changes while typing
    def inputs_active():
        return (x_box.active or y_box.active or speed_box.active
                or angle_box.active or spin_box.active)

    def valid_release_xy(x, y):
        # keep inside side lines and roughly in the throwing zone
        x_min = -W_SHEET/2 + STONE_R
        x_max = +W_SHEET/2 - STONE_R
        y_min = near["hack"] + STONE_R
        y_max = near["hog_inner"] - STONE_R
        x_ok = (x_min <= x <= x_max)
        y_ok = (y_min <= y <= y_max)
        return x_ok and y_ok, (x_min, x_max, y_min, y_max)

    def try_add_stone():
        nonlocal next_team, thrown_red, thrown_yel, error_msg, score_text, x_release, y_release
        if any_moving(stones):
            error_msg = "Wait: stones are still moving."
            return
        if len(stones) >= TOTAL_STONES:
            error_msg = "End full: no more stones."
            return

        # Read inputs
        xv, ok_x = x_box.get_value()
        yv, ok_y = y_box.get_value()
        v, ok_v = speed_box.get_value()
        ang_deg, ok_a = angle_box.get_value()
        omg, ok_o = spin_box.get_value()

        if not (ok_x and ok_y and ok_v and ok_a and ok_o):
            error_msg = "Please enter valid numbers for all fields."
            return

        # Range checks for X,Y
        x_ok, (x_min, x_max, y_min, y_max) = valid_release_xy(xv, yv)
        if not x_ok:
            error_msg = f"X,Y out of range. X∈[{x_min:.2f},{x_max:.2f}], Y∈[{y_min:.2f},{y_max:.2f}]"
            return

        # Clamp speed/spin
        v = max(0.1, min(4.5, float(v)))
        psi = math.radians(float(ang_deg))
        omega = max(-12.0, min(12.0, float(omg)))

        # Enforce general down-ice direction toward far tee
        to_far = aim_at(xv, yv, 0.0, tee_far)
        dpsi = (psi - to_far + math.pi) % (2*math.pi) - math.pi
        if abs(dpsi) > math.pi/2:
            psi = to_far

        x_release, y_release = xv, yv
        s = Stone(x=x_release, y=y_release, v=v, psi=psi, omega=omega)
        setattr(s, "_team", next_team)
        stones.append(s)
        if next_team == 0:
            thrown_red += 1
        else:
            thrown_yel += 1
        next_team = 1 - next_team
        score_text = ""
        error_msg = ""

    error_msg = ""
    score_text = ""
    running = True
    last_any_moving = False
    final_report = ""

    while running:
        dt = clock.tick(fps) / 1000.0

        # ---------- events ----------
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False
            elif evt.type == pygame.VIDEORESIZE:
                width, height = evt.w, evt.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            elif evt.type == pygame.KEYDOWN:
                if inputs_active():
                    # While typing into a field: only Enter should trigger an add.
                    if evt.key == pygame.K_RETURN:
                        try_add_stone()
                else:
                    # No input focused → allow global hotkeys
                    if evt.key == pygame.K_1:
                        view_main = "full"
                    elif evt.key == pygame.K_2:
                        view_main = "house_near"
                    elif evt.key == pygame.K_3:
                        view_main = "house_far"
                    elif evt.key == pygame.K_RETURN:
                        if not any_moving(stones):
                            winner, sc, _ = house_score(stones, tee_far)
                            score_text = f"Score (far): {winner or 'None'} {sc}"
                    elif evt.key == pygame.K_r:
                        stones.clear()
                        next_team = 0
                        thrown_red = 0
                        thrown_yel = 0
                        score_text = ""
                        error_msg = ""
                        final_report = ""
                    elif evt.key == pygame.K_z:
                        if stones and not any_moving(stones):
                            last = stones.pop()
                            if getattr(last, "_team", 0) == 0:
                                thrown_red = max(0, thrown_red - 1)
                            else:
                                thrown_yel = max(0, thrown_yel - 1)
                            next_team = (next_team + 1) % 2
                            score_text = ""
                            final_report = ""
                    elif evt.key == pygame.K_TAB:
                        next_team = 1 - next_team

            # Inputs (also catch Enter per-box)
            for box in (x_box, y_box, speed_box, angle_box, spin_box):
                enter_hit = box.handle_event(evt)
                if enter_hit == "__ENTER__":
                    try_add_stone()

            # Button click
            if btn_add.handle_event(evt):
                try_add_stone()

        # ---------- physics ----------
        # Capture "moving → still" transitions across this frame
        last_any_moving = any_moving(stones)

        for s in stones:
            update_stone_motion(s, dt)

        # collisions
        n = len(stones)
        for i in range(n):
            for j in range(i + 1, n):
                if detect_collision(stones[i], stones[j]):
                    resolve_collision(stones[i], stones[j])

        # Did we just come to rest?
        now_any_moving = any_moving(stones)
        if last_any_moving and not now_any_moving and stones:
            s_end = stones[-1]  # newest stone
            _, tee_far_y = tee_positions()
            dist_to_far_tee = math.hypot(s_end.x - 0.0, s_end.y - tee_far_y)
            final_report = (f"Final: x={s_end.x:.3f} m, y={s_end.y:.3f} m  "
                            f"(to far tee center: {dist_to_far_tee:.3f} m)")
            print(final_report)

        # auto-score when end over
        end_full = (len(stones) >= TOTAL_STONES)
        if end_full and not now_any_moving and not score_text:
            winner, sc, _ = house_score(stones, tee_far)
            score_text = f"End over. Score (far): {winner or 'None'} {sc}"

        # ---------- draw ----------
        screen.fill(BG_OUTSIDE)

        # Left = main (fixed "full" unless user hits 1/2/3)
        main_world  = compute_view(view_main)
        # Middle = far house zoom
        panel_world = compute_view("house_far")

        (main_surf, main_rect,
         mid_surf,  mid_rect,
         right_surf,right_rect,
         (m_scale, m_ox, m_oy),
         (p_scale, p_ox, p_oy)) = layout_three_panels(
            screen, main_world, panel_world, gap_px=10, pad_px=10, right_w_px=360, mid_w_px=460
        )

        # Left: main sheet
        main_surf.fill((235, 242, 255))
        draw_sheet(main_surf, main_world, m_scale, m_ox, m_oy)
        draw_lines(main_surf, m_scale, m_ox, m_oy, main_world)
        for s in stones:
            color = HANDLE_RED if getattr(s, "_team", 0) == 0 else HANDLE_YEL
            draw_stone(main_surf, m_scale, m_ox, m_oy, s, color)

        # Crosshair at current release point (reference)
        rx, ry = w2s(x_release, y_release, m_scale, m_ox, m_oy)
        pygame.draw.circle(main_surf, (30, 80, 160), (rx, ry), 4)

        # Middle: far house panel
        mid_surf.fill((235, 242, 255))
        draw_sheet(mid_surf, panel_world, p_scale, p_ox, p_oy)
        draw_lines(mid_surf, p_scale, p_ox, p_oy, panel_world)
        for s in stones:
            color = HANDLE_RED if getattr(s, "_team", 0) == 0 else HANDLE_YEL
            draw_stone(mid_surf, p_scale, p_ox, p_oy, s, color)

        # Box outline + label
        pygame.draw.rect(screen, (160, 170, 180), mid_rect, 2)
        screen.blit(font.render("TARGET HOUSE (far end)", True, HUD_TXT), (mid_rect.x + 8, mid_rect.y + 6))

        # Right: input column
        right_surf.fill((248, 250, 255))
        pygame.draw.rect(screen, (160, 170, 185), right_rect, 2, border_radius=16)

        col_pad = 16
        y0 = right_rect.y + 16
        x0 = right_rect.x + col_pad
        w_field = right_rect.width - 2*col_pad

        # Title
        screen.blit(font_big.render("Throw Inputs", True, (35, 45, 65)), (x0, y0))
        y0 += 36

        # Fields
        h_field = 36
        gap_y = 10

        x_box.rect     = pygame.Rect(x0, y0, w_field, h_field); x_box.draw(screen, font); y0 += h_field + gap_y
        y_box.rect     = pygame.Rect(x0, y0, w_field, h_field); y_box.draw(screen, font); y0 += h_field + gap_y
        speed_box.rect = pygame.Rect(x0, y0, w_field, h_field); speed_box.draw(screen, font); y0 += h_field + gap_y
        angle_box.rect = pygame.Rect(x0, y0, w_field, h_field); angle_box.draw(screen, font); y0 += h_field + gap_y
        spin_box.rect  = pygame.Rect(x0, y0, w_field, h_field); spin_box.draw(screen, font);  y0 += h_field + 14

        btn_add.rect   = pygame.Rect(x0, y0, w_field, 42);      btn_add.draw(screen, font_big); y0 += 52

        # Hints
        hints = [
            "Frame: x=0 centerline, y=0 near tee (to far tee is +y).",
            "ψ (deg): 90 = straight to far tee.",
            "ω (rad/s): spin rate.",
            f"Release X range: [{-W_SHEET/2+STONE_R:.2f}, {W_SHEET/2-STONE_R:.2f}] m",
            f"Release Y range: [{near['hack']+STONE_R:.2f}, {near['hog_inner']-STONE_R:.2f}] m",
            "Enter in a field = throw • 1/2/3 change left view • Z undo • R reset",
        ]
        for h in hints:
            screen.blit(font.render(h, True, (70, 85, 110)), (x0, y0))
            y0 += 18

        if final_report:
            screen.blit(font.render(final_report, True, (40, 120, 50)), (x0, y0))
            y0 += 20

        # HUD bottom left
        moving = any_moving(stones)
        fps_txt = f"{clock.get_fps():.0f}"
        hud_lines = [
            f"{title} | view:{view_main} | fps:{fps_txt}",
            f"Throws: Red {thrown_red}/{MAX_STONES_PER_TEAM} | Yellow {thrown_yel}/{MAX_STONES_PER_TEAM} | Next: {'Red' if next_team==0 else 'Yellow'}",
            f"{'Stones moving...' if moving else 'All still'}  |  {score_text or ''}",
        ]
        yy = 6
        for line in hud_lines:
            screen.blit(font.render(line, True, HUD_TXT), (8, yy))
            yy += 18

        # Error message (if any)
        if error_msg:
            screen.blit(font.render(error_msg, True, (255, 110, 110)), (8, yy + 4))

        pygame.display.flip()

    pygame.quit()

# ---------------- run demo with your measured shot ----------------
if __name__ == "__main__":
    # Shot #1 (your measured values)
    # Position x  = -0.415 m  (left of centerline)
    # Position y  = 14.011 m  (down-ice from near tee)
    # Speed       = 1.6145 m/s
    # Angle       = 87.8236°  (deg; 90° points straight toward far tee)
    # Spin rate   = 1.48 rad/s
    ''''''
    run_pygame_rink()
