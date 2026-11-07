"""144-frame timeline (24 fps). Everything is a pure function of frame f.

  1-30    ride up the kicker, crouching lower and lower
  31-75   launch, swing 90 deg into a boardslide on the neon rail (sparks)
  76-110  pop off the rail end: 360 kickflip + quarter varial, legs tucked
  111-125 land, powerslide to a stop on the right third, turn to camera
  126-144 idle; blink squashes the eyes on Z from 132, open again by 138
"""

import math

from mathutils import Matrix, Vector

from board import DECK_T, DECK_TOP, DECK_Z, WHEEL_R
from lib import clamp, damped, ease_in_out, ease_out, lerp, mat4, quat, seg
from set import RAIL_R, RAIL_X0, RAIL_X1, RAIL_Z, RAMP_H, RAMP_X1, ramp_slope, ramp_z
from spider import STANCE, TUCK

FPS = 24
FRAMES = 144
SP = 1.7  # spider scale relative to the board

F_LIP, F_RAIL, F_POP, F_LAND, F_STOP = 30.5, 42.0, 75.5, 110.5, 125.0
F_BLINK0, F_BLINK_SHUT, F_BLINK1 = 132.0, 134.5, 138.0

X_START = -46.0
V_RIDE = (RAMP_X1 - X_START) / (F_LIP - 1)  # units / frame
X_RAIL0, X_RAIL1 = RAIL_X0 + 0.5, RAIL_X1 - 0.5
V_G0, V_G1 = 0.88, 2 * (X_RAIL1 - X_RAIL0) / (F_POP - F_RAIL) - 0.88
V_AIR = 0.6
X_LAND = X_RAIL1 + V_AIR * (F_LAND - F_POP)
X_STOP = X_LAND + V_AIR * (F_STOP - F_LAND) / 2
Z_SLIDE = RAIL_Z + RAIL_R - DECK_Z  # board origin height while the deck rides the rail
PEAK1, PEAK2 = 6.8, 14.0


def ballistic(z0, z1, zp, T):
    a, b = zp - z0, zp - z1
    g = 2 * (math.sqrt(a) + math.sqrt(b)) ** 2 / (T * T)
    return math.sqrt(2 * g * a), g


VZ1, G1 = ballistic(RAMP_H, Z_SLIDE, PEAK1, F_RAIL - F_LIP)
VZ2, G2 = ballistic(Z_SLIDE, 0.0, PEAK2, F_LAND - F_POP)


def board_xz(f):
    """Board origin x, z (no rotation) at frame f."""
    if f < F_LIP:
        x = X_START + V_RIDE * (f - 1)
        return x, ramp_z(x)
    if f < F_RAIL:
        u = f - F_LIP
        x = RAMP_X1 + (X_RAIL0 - RAMP_X1) * u / (F_RAIL - F_LIP)
        return x, RAMP_H + VZ1 * u - G1 * u * u / 2
    if f < F_POP:
        u = f - F_RAIL
        a = (V_G1 - V_G0) / (F_POP - F_RAIL)
        return X_RAIL0 + V_G0 * u + a * u * u / 2, Z_SLIDE
    if f < F_LAND:
        u = f - F_POP
        return X_RAIL1 + V_AIR * u, Z_SLIDE + VZ2 * u - G2 * u * u / 2
    u = min(f, F_STOP) - F_LAND
    T = F_STOP - F_LAND
    return X_LAND + V_AIR * u - V_AIR / (2 * T) * u * u, 0.0


def wheel_angle(f):
    """Rolled distance / radius, frozen spin while airborne or sliding."""
    def rolled(ff):
        if ff < F_LIP:
            return board_xz(ff)[0] - X_START
        return RAMP_X1 - X_START + V_RIDE * 6 * (1 - math.exp(-(ff - F_LIP) / 6))
    base = rolled(min(f, F_POP))
    if f >= F_LAND:
        base += board_xz(f)[0] - X_LAND
    return base / WHEEL_R


def grinding(f):
    return F_RAIL <= f < F_POP


def contact_point(f):
    x, _ = board_xz(f)
    return Vector((x, 0.0, RAIL_Z + RAIL_R))


def state(f, cam_end):
    """-> board matrix, spider matrix, 8 foot targets, wheel angle, blink."""
    x, z = board_xz(f)
    yaw = pitch = flip = roll = 0.0
    if f < F_LIP:
        pitch = ramp_slope(x)
    elif f < F_RAIL:
        p = (f - F_LIP) / (F_RAIL - F_LIP)
        pitch = ramp_slope(RAMP_X1 - 1e-3) * (1 - ease_out(seg(p, 0, 0.5)))
        yaw = math.pi / 2 * ease_in_out(seg(p, 0.15, 0.95))
    elif f < F_POP:
        u = f - F_RAIL
        yaw = math.pi / 2 + 0.06 * math.sin(u * 0.5)
        roll = 0.05 * damped(u, 0.9, 0.15)  # wobble on impact
    elif f < F_LAND:
        p = (f - F_POP) / (F_LAND - F_POP)
        yaw = math.pi / 2 * (1 - ease_in_out(seg(p, 0.1, 0.8)))
        flip = 2 * math.pi * ease_in_out(seg(p, 0.12, 0.72))
        pitch = 0.18 * math.sin(math.pi * seg(p, 0, 0.35)) - 0.08 * math.sin(math.pi * seg(p, 0.8, 1))
    else:
        u = f - F_LAND
        yaw = -0.45 * ease_in_out(seg(u, 1, F_STOP - F_LAND + 2))
        z += 0.15 * max(0.0, damped(u, 0.9, 0.35))

    cz = DECK_Z + DECK_T / 2
    B0 = mat4((x, 0, z), quat(yaw, pitch, roll))
    B = B0 @ Matrix.Translation((0, 0, cz)) @ Matrix.Rotation(flip, 4, "X") @ Matrix.Translation((0, 0, -cz))

    # ---- spider relative to the unflipped board frame
    crouch = lift = tuck = lean = 0.0
    look_yaw, look_pitch = 0.55, 0.0
    if f < F_LIP:
        crouch = 0.15 + 0.3 * ease_in_out(seg(f, 1, F_LIP - 1))
        crouch += 0.04 * math.sin(f * 0.6)
    elif f < F_RAIL:
        p = (f - F_LIP) / (F_RAIL - F_LIP)
        crouch = 0.45 * (1 - ease_out(seg(p, 0, 0.3))) + 0.3 * seg(p, 0.7, 1)
        lift = 0.6 * math.sin(math.pi * p)
        look_yaw = lerp(0.55, 0.0, ease_in_out(p))
    elif f < F_POP:
        u = f - F_RAIL
        crouch = 0.3 * max(0.0, damped(u + 1, 0.7, 0.2)) + 0.18
        lean = 0.1 * math.sin(u * 0.45)
        look_yaw = 0.0
        crouch += 0.2 * ease_in_out(seg(f, F_POP - 4, F_POP))
    elif f < F_LAND:
        p = (f - F_POP) / (F_LAND - F_POP)
        lift = 2.2 * math.sin(math.pi * seg(p, 0.0, 0.9)) ** 0.7
        tuck = math.sin(math.pi * seg(p, 0.02, 0.88))
        look_yaw = lerp(0.0, 0.3, p)
        look_pitch = -0.15 * math.sin(math.pi * p)  # watches the board flip
    else:
        u = f - F_LAND
        crouch = 0.5 * max(0.0, damped(u + 0.5, 0.5, 0.2)) + 0.1 * (1 - seg(u, 0, 10))
        lean = 0.22 * ease_in_out(seg(u, 1, 7)) * (1 - ease_in_out(seg(u, 9, 15)))
        look_yaw = 0.3

    # turn toward the lens during the slide, then hold the hero stare
    k = ease_in_out(seg(f, F_LAND + 4, F_STOP + 2))
    cam_b = B0.inverted() @ Vector(cam_end)
    face = math.atan2(cam_b.y, cam_b.x) + math.pi / 2
    look_yaw = lerp(look_yaw, face, k) + 0.05 * damped(f - F_STOP - 2, 0.6, 0.2)
    look_pitch = lerp(look_pitch, 0.2, k)
    if f >= F_STOP:
        crouch -= 0.06 * math.sin((f - F_STOP) * 0.35)  # idle breathing bob

    h = (1.0 - crouch) * SP + lift
    S_rel = mat4((0.2, 0, DECK_TOP + h), quat(-math.pi / 2 + look_yaw, look_pitch, lean))
    S_rel = S_rel @ Matrix.Diagonal((SP, SP, SP, 1))
    S = B0 @ S_rel

    feet = []
    for i in range(4):
        for side in (1, -1):
            fx = lerp(STANCE[i][0], TUCK[i][0], tuck)
            fy = lerp(STANCE[i][1], TUCK[i][1], tuck) * side
            if tuck > 0.02:
                feet.append(S @ Vector((fx, fy, -0.8 - 0.25 * (1 - tuck))))
            else:
                # feet stay planted on the deck ("parented" to the board)
                loc = S_rel @ Vector((fx, fy, 0))
                bx = clamp(loc.x, -4.2, 4.2)
                by = clamp(loc.y, -1.25, 1.25)
                feet.append(B0 @ Vector((bx, by, DECK_TOP + 0.02)))

    b = 0.0
    if F_BLINK0 <= f < F_BLINK1:
        b = ease_in_out(seg(f, F_BLINK0, F_BLINK_SHUT)) if f < F_BLINK_SHUT else 1 - ease_in_out(seg(f, F_BLINK_SHUT + 0.5, F_BLINK1))
    return B, S, feet, wheel_angle(f), b
