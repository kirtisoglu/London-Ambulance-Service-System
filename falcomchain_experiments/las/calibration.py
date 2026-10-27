"""
Locked calibration of the London Ambulance Service instance (paper, Section 7.4).

The capacity unit is a block of k = 3 ambulances; w_unit = 10,887 calls per
unit per year (five patients per shift, two shifts per day, 365 days, times a
block of three ambulances, rounded so that ceil(d / w) = 97 units cover the
2010-2015 demand and the terminal debt of the recursion is negligible); eps^1 = eps^2 =
0.15; c^1 in [1, 3] units, c^2 in [2, 6] units; every super-district holds at
least kappa^2_min = 2 districts; gamma^1 = gamma^2 = 0 (cut selection uniform
over admissible cuts).
"""

W = 10_887                          # calls per unit (3 ambulances) per year
EPS_BASE = 0.15
EPS_SUPER = 0.15
C_MIN_BASE = 1
C_MAX_BASE = 3
C_MIN_SUPER = 2 * C_MIN_BASE        # 2 units
C_MAX_SUPER = 2 * C_MAX_BASE        # 6 units
MIN_DISTRICTS_SUPER = 2             # kappa^2_min
GAMMA_BASE = 0.0
GAMMA_SUPER = 0.0
