import math
import numpy as np
from dataclasses import dataclass, field

# Tunable physical constants (defaults: "faster ice / livelier hits") 
mu_ice   = 0.0135  # ice friction coefficient (a = mu_ice * g). Lower -> farther glide.
g        = 9.81    # gravity (m/s^2)
kappa    = 1.2     # curl factor (spin → heading change rate)
gamma    = 0.3     # spin damping rate (rad/s^2). Lower -> spin lasts longer.
M        = 20.0    # stone mass (kg)
D        = 0.29    # stone diameter (m)
EPS_V    = 1e-6    # minimal speed threshold for curl computation

# Collision constants (tuned to push struck stones farther) 
e_coll      = 0.98   # restitution (normal). Higher -> livelier bounce/transfer.
mu_coll     = 0.08   # stone-on-stone tangential friction. Lower -> less energy loss to spin/sideways.
spin_factor = 0.8    # scaling of spin contribution to tangential slip. Lower -> less linear→spin drain.
slip_eps    = 5e-3   # slip vs stick threshold (bigger -> more slipping, less tangential energy loss)

#  Geometry and inertia 
R = D / 2.0
contact_R = 0.065
I  = 0.15

#
# We choose: positive omega = CLOCKWISE (CW). In math coordinates, positive rotation is CCW,
# so we flip the sign wherever omega couples to heading/relative slip/visualization.
SPIN_POSITIVE_IS_CW = True
_spin_sign = -1.0 if SPIN_POSITIVE_IS_CW else 1.0

# ---- Small helpers for calibration / live tuning ----
def set_tuning(ice=None, e=None, mu=None, spinfac=None, slip=None, gamma_=None, kappa_=None):
    """Change model parameters at runtime."""
    global mu_ice, e_coll, mu_coll, spin_factor, slip_eps, gamma, kappa
    if ice     is not None: mu_ice = float(ice)
    if e       is not None: e_coll = float(e)
    if mu      is not None: mu_coll = float(mu)
    if spinfac is not None: spin_factor = float(spinfac)
    if slip    is not None: slip_eps = float(slip)
    if gamma_  is not None: gamma = float(gamma_)
    if kappa_  is not None: kappa = float(kappa_)

def estimate_mu_from_shot(v0, stop_distance):
    """
    Estimate mu_ice from a no-collision shot assuming constant decel:
    s_stop = v0^2 / (2*mu*g)  ->  mu = v0^2 / (2*g*s_stop)
    """
    return (v0*v0) / (2.0 * g * max(1e-6, stop_distance))


class Stone:
    """Represents a curling stone in 2D with position, velocity, and spin."""
    def __init__(self, x, y, v, psi, omega, team=None, phi=0.0, moving=True):
        self.x = x
        self.y = y
        self.v = v
        self.psi = psi
        self.omega = omega
        self.moving = moving
        self.phi = phi
        self.xs = []
        self.ys = []
        self.team = team

    def log(self):
        self.xs.append(self.x)
        self.ys.append(self.y)


def update_stone_motion(stone: Stone, dt: float):
    """Advance a stone by dt via ice friction, spin decay, curl, and translation."""
    if not stone.moving:
        return

    # ice friction deceleration (linear model)
    stone.v = max(stone.v - mu_ice * g * dt, 0.0)

    # spin decay
    stone.omega -= gamma * stone.omega * dt

    # curl: heading changes with spin (positive omega = CW -> curl right, i.e., decrease psi)
    if stone.v > EPS_V:
        stone.psi += (kappa / M) * (_spin_sign * stone.omega / stone.v) * dt

    # translate
    stone.x += stone.v * math.cos(stone.psi) * dt
    stone.y += stone.v * math.sin(stone.psi) * dt

    # track spin phase for visualization (match our CW-positive convention)
    stone.phi += _spin_sign * stone.omega * dt

    # stop if very slow
    if stone.v < 1e-3:
        stone.v = 0.0
        stone.moving = False


def detect_collision(a: Stone, b: Stone) -> bool:
    """Return True if stones are touching or overlapping."""
    if not (a.moving or b.moving):
        return False
    dx, dy = a.x - b.x, a.y - b.y
    return (dx*dx + dy*dy) <= (D * D)


def resolve_collision(a: Stone, b: Stone):
    """
    Nearly elastic + reduced tangential loss collision between equal mass stones,
    exchanging both normal and tangential impulses and spin.
    """
    # normal and tangent unit vectors
    dx, dy = a.x - b.x, a.y - b.y
    dist_sq = dx*dx + dy*dy
    if dist_sq == 0:
        return
    dist = math.sqrt(dist_sq)
    nx, ny = dx/dist, dy/dist
    tx, ty = -ny, nx

    # rewind overlap so collision triggers at contact
    overlap = D - dist
    if overlap > 0:
        if a.moving and not b.moving:
            a.x += nx * overlap
            a.y += ny * overlap
        elif b.moving and not a.moving:
            b.x -= nx * overlap
            b.y -= ny * overlap
        dx, dy = a.x - b.x, a.y - b.y
        dist = math.sqrt(dx*dx + dy*dy)
        nx, ny = dx/dist, dy/dist
        tx, ty = -ny, nx

    # world velocities
    v1x, v1y = a.v * math.cos(a.psi), a.v * math.sin(a.psi)
    v2x, v2y = b.v * math.cos(b.psi), b.v * math.sin(b.psi)

    # decompose
    v1n = v1x*nx + v1y*ny
    v1t = v1x*tx + v1y*ty
    v2n = v2x*nx + v2y*ny
    v2t = v2x*tx + v2y*ty

    # normal impulse (elastic-ish)
    Jn = -(1 + e_coll) * (v1n - v2n) * (M/2)

    # relative tangential slip with running band radius
    # Apply same CW-positive convention here
    v_rel_t = (v1t - v2t) + spin_factor * contact_R * (_spin_sign * (a.omega + b.omega))

    # tangential impulse, capped by Coulomb
    slipping = abs(v_rel_t) > slip_eps
    if slipping:
        Jt_uncapped = -v_rel_t * (M/2)
        Jt = max(-mu_coll*abs(Jn), min(mu_coll*abs(Jn), Jt_uncapped))
    else:
        Jt_stick = (v2t - v1t - spin_factor * contact_R * (_spin_sign * (a.omega + b.omega))) * (M/2)
        Jt = max(-mu_coll*abs(Jn), min(mu_coll*abs(Jn), Jt_stick))

    # apply impulses (linear)
    v1n_new = v1n + Jn/M
    v2n_new = v2n - Jn/M
    v1t_new = v1t + Jt/M
    v2t_new = v2t - Jt/M

    # world velocities after
    v1x_new = v1n_new*nx + v1t_new*tx
    v1y_new = v1n_new*ny + v1t_new*ty
    v2x_new = v2n_new*nx + v2t_new*tx
    v2y_new = v2n_new*ny + v2t_new*ty

    # torque from tangential impulse
    Fx_t = Jt * tx
    Fy_t = Jt * ty
    rx, ry = nx * contact_R, ny * contact_R
    torqueA = rx * Fy_t - ry * Fx_t
    torqueB = -torqueA

    # spins
    a.omega += torqueA / I
    b.omega += torqueB / I

    # speeds/headings
    a.v = math.hypot(v1x_new, v1y_new)
    b.v = math.hypot(v2x_new, v2y_new)
    if a.v > EPS_V:
        a.psi = math.atan2(v1y_new, v1x_new)
    if b.v > EPS_V:
        b.psi = math.atan2(v2y_new, v2x_new)

    a.moving = (a.v > 1e-3)
    b.moving = (b.v > 1e-3)


def out_detected(stone: Stone, xlim: tuple, ylim: tuple) -> bool:
    """Return True if stone is out of bounds."""
    return not (xlim[0] <= stone.x <= xlim[1] and ylim[0] <= stone.y <= ylim[1])


def simulate(stones, dt=0.001, t_max=10.0):
    t = 0.0
    steps = 0
    out_stones = [False] * len(stones)

    button = (0.0, 34.747)
    sheet_width = 4.75
    xlim = (button[0] - sheet_width / 2.0, button[0] + sheet_width / 2.0)
    ylim = (1.37, button[1] + 1.829)

    while t < t_max and any(s.moving for s in stones):
        for s in stones:
            update_stone_motion(s, dt)

        # collisions
        n = len(stones)
        for i in range(n):
            for j in range(i+1, n):
                if detect_collision(stones[i], stones[j]):
                    resolve_collision(stones[i], stones[j])

        for i, s in enumerate(stones):
            s.log()
            if out_detected(s, xlim=xlim, ylim=ylim):
                s.moving = False
                out_stones[i] = True

        t += dt
        steps += 1

    return steps, out_stones


if __name__ == "__main__":
    # Example: estimate mu from a single no-collision shot
    v0 = 1.6145
    s_stop_measured = 8.0  # <-- put your measured travel (m) here to see mu estimate
    print("mu_ice estimate:", estimate_mu_from_shot(v0, s_stop_measured))

    # quick two-stone sanity
    s1 = Stone(x=0, y=1.37, v=3.4, psi=np.radians(110), omega=5.0)  # +omega is CW now
    simulate([s1], dt=0.001, t_max=8.0)
    button = (0.0, 34.747)
    print(s1.x, s1.y)
    print("Distance to button:", math.hypot(s1.x - button[0], s1.y - button[1]))
