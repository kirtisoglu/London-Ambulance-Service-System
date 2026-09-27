"""
Build the v2 LAS instance data from authoritative LAS sources:

1. ``data/derived/lsoa_to_sector_borough.csv`` — LSOA → sector by NHS
   ICS / LAS administrative borough membership (the official 5-sector
   layout, no Voronoi approximation).  See FiveSectorMap (FOI 6055,
   2018) page 1 right-hand-panel borough lists.
2. ``data/derived/sectors_polygons.gpkg`` — 5 dissolved borough
   polygons by sector (for figures, point-in-polygon checks).
3. ``data/raw/LAS_L2_facilities.csv`` (v2) — replaces the 5
   "geographically central station" proxies with the **21 Group HQs**
   from LAS-List-Of-Departments-2025.pdf, plus the 2 EOCs (Waterloo,
   Newham).  Snapped to LSOA centroids using the existing
   LAS_stations.csv coordinates.
4. ``data/derived/las_fleet_per_station.csv`` — per-Group-HQ
   operational vehicle counts (DCAs, FRUs, MRUs, etc.) from
   LAS-Fleet-April-2025.pdf.

Run:
    python analysis/run_build_v2_data.py
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pdfplumber

REPO = Path(__file__).resolve().parents[1]
LSOA_GPKG = REPO / "data/raw/LSOA_2021_London.gpkg"
STATIONS_CSV = REPO / "data/raw/LAS_stations.csv"
FLEET_PDF = REPO / "new_data/LAS-Fleet-April-2025.pdf"
FLEET_PDF_FOI7376 = REPO / "new_data/Appendix 1 FOI 7376 Fleet List.pdf"
# Operational-to-inventory ratio.  LAS public optimisation literature
# (Coates & McCormack 2015; FOI 7198) reports ~446 emergency ambulances
# *on shift* against an inventory ~600 → factor 0.706.  Used to derive
# the per-Group "operational team" count from the registered DCA count.
ON_SHIFT_RATIO = 0.706

OUT_BOROUGH_SECTOR = REPO / "data/derived/lsoa_to_sector_borough.csv"
OUT_SECTOR_POLYS = REPO / "data/derived/sectors_polygons.gpkg"
OUT_L2_FACILITIES = REPO / "data/raw/LAS_L2_facilities.csv"
# LAS-Fleet-April-2025.pdf only reports vehicle counts at the Group HQ
# station name (no row exists for non-HQ stations like Poplar, Silvertown,
# etc.).  We keep the data at Group granularity — the FalCom energy
# functions (E_minisum, E_fair) do not use capacity, so per-L1-station
# vehicle counts are not on the critical path; reporting imbalance at
# the super-district (Group) level is the most honest reading of the
# source data.
OUT_FLEET_PER_GROUP = REPO / "data/derived/las_fleet_per_group.csv"

# -----------------------------------------------------------------------------
# 1) BOROUGH → SECTOR.  Source: FiveSectorMap (FOI 6055, 2018), page 1.
#    The 5 LAS sectors correspond exactly to NHS England's 5 London ICS
#    boundaries (NCL / NEL / NWL / SEL / SWL).
# -----------------------------------------------------------------------------
BOROUGH_TO_SECTOR = {
    # North Central
    "Camden": "North Central",
    "Barnet": "North Central",
    "Enfield": "North Central",
    "Haringey": "North Central",
    "Islington": "North Central",
    # North East (includes City of London via City & Hackney CCG)
    "Barking and Dagenham": "North East",
    "City of London": "North East",
    "Hackney": "North East",
    "Havering": "North East",
    "Newham": "North East",
    "Redbridge": "North East",
    "Tower Hamlets": "North East",
    "Waltham Forest": "North East",
    # North West (NWL: Brent, Ealing, H&F, Harrow, Hillingdon, Hounslow, K&C, Westminster)
    "Brent": "North West",
    "Ealing": "North West",
    "Hammersmith and Fulham": "North West",
    "Harrow": "North West",
    "Hillingdon": "North West",
    "Hounslow": "North West",
    "Kensington and Chelsea": "North West",
    "Westminster": "North West",
    # South East (SEL: Bexley, Bromley, Greenwich, Lambeth, Lewisham, Southwark)
    "Bexley": "South East",
    "Bromley": "South East",
    "Greenwich": "South East",
    "Lambeth": "South East",
    "Lewisham": "South East",
    "Southwark": "South East",
    # South West (SWL: Croydon, Kingston, Merton, Richmond, Sutton, Wandsworth)
    "Croydon": "South West",
    "Kingston upon Thames": "South West",
    "Merton": "South West",
    "Richmond upon Thames": "South West",
    "Sutton": "South West",
    "Wandsworth": "South West",
    # Brentwood is in Essex (outside LAS); kept for completeness only.
    "Brentwood": "Outside LAS",
}

# -----------------------------------------------------------------------------
# 2) GROUP HQ table.  Source: LAS-List-Of-Departments-2025.pdf "ECS" branch.
#    Each Group is named after the lead station that hosts the Group Manager
#    (the "Group HQ").  21 groups across 5 sectors.
# -----------------------------------------------------------------------------
GROUP_HQS = [
    # (station_name_in_LAS_stations.csv, sector, group_label)
    ("Camden",        "North Central", "Camden Group"),
    ("Edmonton",      "North Central", "Edmonton Group"),
    ("Friern Barnet", "North Central", "Friern Barnet Group"),
    ("Homerton",      "North East",    "Homerton Group"),
    ("Ilford",        "North East",    "Ilford Group"),
    ("Newham",        "North East",    "Newham Group"),
    ("Romford",       "North East",    "Romford Group"),
    ("Whipps Cross",  "North East",    "Whipps Cross Group"),
    ("Brent",         "North West",    "Brent Group"),
    ("Fulham",        "North West",    "Fulham Group"),
    ("Hanwell",       "North West",    "Hanwell Group"),
    # Hillingdon Group HQ — the Hillingdon station (B5) is now in the
    # roster (added from FOI 7353), so it acts as its own Group HQ.
    ("Hillingdon",    "North West",    "Hillingdon Group"),
    ("Westminster",   "North West",    "Westminster Group"),
    ("Bromley",       "South East",    "Bromley Group"),
    ("Deptford",      "South East",    "Deptford Group"),
    ("Greenwich",     "South East",    "Greenwich Group"),
    ("Oval",          "South East",    "Oval Group"),
    ("Croydon",       "South West",    "Croydon Group"),
    ("New Malden",    "South West",    "New Malden Group"),
    ("St Helier",     "South West",    "St Helier Group"),
    ("Wimbledon",     "South West",    "Wimbledon Group"),
]


def build_borough_sector_csv() -> None:
    """Write per-LSOA sector assignment derived from borough membership."""
    lsoa = gpd.read_file(LSOA_GPKG)
    # The LSOA name encodes the borough (everything before the trailing " NNNL").
    import re
    lsoa_suffix = re.compile(r" \d{3}[A-Z]$")
    lsoa["borough"] = lsoa["LSOA21NM"].str.replace(lsoa_suffix, "", regex=True)
    missing = sorted(set(lsoa["borough"]) - set(BOROUGH_TO_SECTOR))
    if missing:
        raise RuntimeError(f"Boroughs not in lookup: {missing}")
    lsoa["sector"] = lsoa["borough"].map(BOROUGH_TO_SECTOR)
    out = lsoa[["LSOA21CD", "LSOA21NM", "borough", "sector"]].copy()
    out.to_csv(OUT_BOROUGH_SECTOR, index=False)
    by_sec = out.groupby("sector").size().sort_values(ascending=False)
    print(f"[1/4] wrote {OUT_BOROUGH_SECTOR.relative_to(REPO)} ({len(out)} LSOAs)")
    for sec, n in by_sec.items():
        print(f"        {sec:15s}  {n} LSOAs")


def dissolve_sector_polygons() -> None:
    """Dissolve borough polygons by sector → 5 super-polygons."""
    lsoa = gpd.read_file(LSOA_GPKG)
    import re
    lsoa_suffix = re.compile(r" \d{3}[A-Z]$")
    lsoa["borough"] = lsoa["LSOA21NM"].str.replace(lsoa_suffix, "", regex=True)
    lsoa["sector"] = lsoa["borough"].map(BOROUGH_TO_SECTOR)
    sectors = lsoa.dissolve(by="sector").reset_index()
    # Drop the "Outside LAS" Brentwood pocket from the LAS polygon GeoPackage.
    sectors = sectors[sectors["sector"] != "Outside LAS"].copy()
    sectors = sectors[["sector", "geometry"]]
    sectors.to_file(OUT_SECTOR_POLYS, driver="GPKG")
    print(f"[2/4] wrote {OUT_SECTOR_POLYS.relative_to(REPO)} "
          f"({len(sectors)} sector polygons)")
    for _, r in sectors.iterrows():
        area_km2 = r.geometry.area / 1e6  # geometry is BNG (m); /1e6 → km²
        print(f"        {r['sector']:15s}  {area_km2:6.1f} km²")


def regenerate_l2_facilities() -> None:
    """Rewrite LAS_L2_facilities.csv with 21 Group HQs + 2 EOCs.

    All Group HQ coordinates are looked up from LAS_stations.csv (the
    Group HQ is by definition a real station).
    """
    stations = pd.read_csv(STATIONS_CSV)
    name_to_row = {n.strip().lower(): r for _, r in stations.iterrows()
                   for n in [r["station_name"]]}
    fac_rows = []
    missing = []
    for sname, sector, glabel in GROUP_HQS:
        st = name_to_row.get(sname.lower())
        if st is None:
            missing.append(sname)
            continue
        fac_rows.append({
            "facility_id":   f"GH-{sname.replace(' ', '_')}",
            "facility_name": f"{glabel} HQ ({sname})",
            "facility_type": "group_hq",
            "LSOA21CD":      st["LSOA21CD"],
            "LSOA21NM":      st["LSOA21NM"],
            "latitude":      st["latitude"],
            "longitude":     st["longitude"],
            "sector":        sector,
            "notes":         "Group HQ from LAS-List-Of-Departments-2025; "
                             "snapped to host station's LSOA centroid.",
        })
    if missing:
        raise RuntimeError(f"Group HQ stations not found in LAS_stations.csv: {missing}")

    # EOCs (unchanged from previous LAS_L2_facilities.csv).
    fac_rows.extend([
        {
            "facility_id":   "EOC-Waterloo",
            "facility_name": "Waterloo EOC",
            "facility_type": "eoc",
            "LSOA21CD":      "E01032582",
            "LSOA21NM":      "Lambeth 036E",
            "latitude":      51.50246849428194,
            "longitude":     -0.1128035347427578,
            "sector":        "",
            "notes":         "LAS Trust HQ, 220 Waterloo Road SE1 8SD.",
        },
        {
            "facility_id":   "EOC-Newham",
            "facility_name": "Newham EOC",
            "facility_type": "eoc",
            "LSOA21CD":      "E01003523",
            "LSOA21NM":      "Newham 024A",
            "latitude":      51.531042925238616,
            "longitude":     0.05664513327204443,
            "sector":        "",
            "notes":         "Co-located with Newham ambulance station.",
        },
    ])

    df = pd.DataFrame(fac_rows)
    df.to_csv(OUT_L2_FACILITIES, index=False)
    print(f"[3/4] wrote {OUT_L2_FACILITIES.relative_to(REPO)} "
          f"({len(df)} L2 candidates: 21 Group HQs + 2 EOCs)")


# -----------------------------------------------------------------------------
# 4) FLEET COUNTS per Group HQ.  Source: LAS-Fleet-April-2025.pdf.
#    "A&E AMBULANCE" = DCA (Double-Crewed Ambulance, our primary "team").
# -----------------------------------------------------------------------------
SKIP_STATUS = {"MUSEUM FLEET", "PRE-FLEET", "HIRE VEHICLE", "Demo Vehicle"}
TRACKED_TYPES = (
    "A&E AMBULANCE",
    "FAST RESPONSE UNIT",
    "MOTOR CYCLE RESPONSE UNIT",
    "CYCLE RESPONSE UNIT",
    "ADVANCE PARAMEDIC",
    "INCIDENT RESPONSE",
    "FIRST RESPONSE",
    "CLINICAL TEAM LEADER",
    "URGENT CARE RESPONSE",
    "MENTAL HEALTH",
)


def _norm_station(dept: str) -> str:
    """Map a Fleet 'Departments' string to its host Group HQ station name.
    Returns '' if the vehicle is non-station-based (HART, LOGISTICS, etc.).
    """
    s = dept.upper().strip()
    # Strip resource-type suffixes.
    for suf in (" MRU", " WSHOP", " WS", " CHUB", " TEAM", " CRU"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    # Strip "TRAINING " prefix.
    if s.startswith("TRAINING "):
        s = s[len("TRAINING ") :]
    if s.startswith("FRU "):
        s = s[len("FRU ") :]
    if s.startswith("MAKE READY"):
        return ""
    if s.startswith("NETS"):
        return ""
    if s.startswith("HART"):
        return ""
    if s in {"CODY ROAD", "TACT.RES.UNIT-EPRR", "WATERLOO CHUB",
             "EMERGENCY OPS", "LOGISTICS", "URGENT CARE APP -",
             "URGENT CARE FRU -", "MENTAL HEALTH -",
             "EMERG PLNNG CODY RD", "FLEET OFFICE", "EMERGENCY PLANNING",
             "TACTICAL RESP.UNIT", "ADMIN & SUPPORT, HQ",
             "EDUCATION & DEVELOP", "EAST HQ - ILFORD", "RESUS",
             "ISLINGTON", "GREENFORD", "PINNER", "CHASE FARM",
             "BARNEHURST", "ISLEWORTH", "KENTON", "RICHMOND",
             "BECONTREE", "STREATHAM", "TOLWORTH", "HEATHROW",
             "COULSDON", "FIRST RESPONSE", "TBA", "1G", "ON DISPLAY",
             "RESUS UNIT POCOCK ST", "MUSEUM KEMSING", "MUSEUM",
             "BOW", "HQ - OPERATIONS", "WELLBEING HUB -",
             "VRC NEWHAM", "VRC ISLEWORTH", "VRC FULHAM",
             "VRC GREENWICH", "VRC WATERLOO", "VRC CROYDON",
             "VRC EDMONTON", "VRC CAMDEN", "VRC", "SILVERTOWN",
             "WELLBEING HUB", "IM&T FIELDEN HOUSE", "IM&T",
             "A&E DIRECTORATE, HQ", "SUTTON", "WANDSWORTH",
             "MAKE READY SW ST", "MAKE READY SW", "MAKE READY SE",
             "MAKE READY NW", "MAKE READY NC", "MAKE READY NW BRENT",
             "WEST HAM", "E-SORT BECKENHAM", "E-SORT", "CRU STRATFORD",
             "MAKE READY NE WEST", "MAKE READY NC FRIERN",
             "MAKE READY NC ILFORD", "MAKE READY BOW",
             "FRIEND BARNET", "TRU - CODY ROAD", "TRU - CLOCK TOWER",
             "CARU (CLINICAL AUDIT &", "ESTATES DEPARTMENT",
             "MAKE READY BARNEHURST HUB",
             "TWICKENHAM", "PRE-FLEET"}:
        return ""
    # Map known group-HQ station names (case-normalised here, then title-cased)
    GROUP_HQ_UPPER = {n.upper() for n, _, _ in GROUP_HQS}
    # Hillingdon needs to pass through here even though it isn't in GROUP_HQS
    # (we alias it to HAYES one level up so the count rolls in correctly).
    return s if s in GROUP_HQ_UPPER else ""


def _extract_apr2025_rows() -> list[list[str]]:
    """Extract per-vehicle rows from LAS-Fleet-April-2025.pdf
    (6-col table: Fleet No., Status, Make, Model, Departments, Vehicle Type)."""
    out = []
    with pdfplumber.open(FLEET_PDF) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables() or []:
                for r in tbl:
                    if r and r[0] and r[0].strip() not in ("Fleet No.", "", None):
                        r = [(c or "").strip() for c in r]
                        if len(r) >= 6:
                            # Return (status, dept, vtype)
                            out.append([r[1], r[4], r[5]])
    return out


def _extract_foi7376_rows() -> list[list[str]]:
    """Extract per-vehicle rows from FOI 7376 Fleet List (no Status column,
    4 cols: Make, Model, Departments, Vehicle Type).  PDF has no table grid;
    parse by word x-position.  All rows here are 'live' (no status field).
    """
    out = []
    import collections as _c
    with pdfplumber.open(FLEET_PDF_FOI7376) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=3)
            lines = _c.defaultdict(list)
            for w in words:
                lines[round(w["top"])].append(w)
            for k in sorted(lines):
                rw = sorted(lines[k], key=lambda w: w["x0"])
                cols = [[], [], [], []]
                for w in rw:
                    x = w["x0"]
                    if x < 120:   cols[0].append(w["text"])
                    elif x < 245: cols[1].append(w["text"])
                    elif x < 395: cols[2].append(w["text"])
                    else:         cols[3].append(w["text"])
                row = [" ".join(c).strip() for c in cols]
                if row[0] and row[0] != "Make":
                    # Mimic 6-col shape: ("Live on fleet", dept, vtype)
                    out.append(["Live on fleet", row[2], row[3]])
    return out


def aggregate_fleet() -> None:
    """Read the LAS-Fleet-April-2025 inventory and write per-Group
    DCA / FRU / MRU / APP / on-shift counts.  Single-source: we use
    only the April-2025 snapshot because it is closest in time to the
    August-2023 sector boundary map (~20 months vs ~31 for FOI 7376).

    Columns:
      n_dca     = A&E AMBULANCEs (DCAs) at the Group HQ
      n_dca_op  = round(n_dca × ON_SHIFT_RATIO)  -- on-shift count
      n_fru     = Fast Response Units
      n_mru     = Motorcycle Response Units
      n_app     = Advanced Paramedic Practitioner cars
    """
    rows_apr = _extract_apr2025_rows()
    rows_foi = _extract_foi7376_rows()
    print(f"        loaded apr={len(rows_apr)} rows, foi7376={len(rows_foi)} rows")
    rows = [("apr", r) for r in rows_apr] + [("foi", r) for r in rows_foi]

    # Hillingdon used to be a Hayes proxy; now that Hillingdon (B5) is
    # its own row in LAS_stations.csv no aliasing is needed.
    FLEET_NAME_ALIAS: dict[str, str] = {}

    by_station = {"apr": collections.defaultdict(lambda: collections.Counter()),
                  "foi": collections.defaultdict(lambda: collections.Counter())}
    for src, r in rows:
        if len(r) < 3:
            continue
        status, dept, vtype = r[0], r[1], r[2]
        if status in SKIP_STATUS:
            continue
        if vtype not in TRACKED_TYPES:
            continue
        host = _norm_station(dept)
        if not host:
            continue
        host = FLEET_NAME_ALIAS.get(host, host)
        by_station[src][host][vtype] += 1

    # Single-source: use only the LAS-Fleet-April-2025 snapshot — it's
    # closer in time to the August-2023 boundary map (~20 months) than
    # the FOI-7376 list (~31 months).  The FOI-7376 extraction stays in
    # the script as a sanity-check counter-source but is not written to
    # the output table.
    out_rows = []
    for sname, sector, glabel in GROUP_HQS:
        ca = by_station["apr"].get(sname.upper(), collections.Counter())
        n_dca = ca["A&E AMBULANCE"]
        n_fru = ca["FAST RESPONSE UNIT"]
        n_mru = ca["MOTOR CYCLE RESPONSE UNIT"]
        n_app = ca["ADVANCE PARAMEDIC"]
        n_op = round(n_dca * ON_SHIFT_RATIO)
        out_rows.append({
            "station_name": sname,
            "group_label":  glabel,
            "sector":       sector,
            "n_dca":        n_dca,         # April 2025 inventory
            "n_dca_op":     n_op,          # on-shift = round(n_dca × 0.706)
            "n_fru":        n_fru,
            "n_mru":        n_mru,
            "n_app":        n_app,
        })
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_FLEET_PER_GROUP, index=False)
    total_inv = df["n_dca"].sum()
    total_op = df["n_dca_op"].sum()
    total_fru = df["n_fru"].sum()
    print(f"[4/4] wrote {OUT_FLEET_PER_GROUP.relative_to(REPO)} "
          f"({len(df)} group HQs)")
    print(f"        DCA inventory (LAS-Fleet-April-2025):  total {total_inv}  "
          f"range [{df['n_dca'].min()}, {df['n_dca'].max()}]  "
          f"mean {df['n_dca'].mean():.1f}")
    print(f"        DCA on-shift (×{ON_SHIFT_RATIO}):       total {total_op}  "
          f"range [{df['n_dca_op'].min()}, {df['n_dca_op'].max()}]  "
          f"mean {df['n_dca_op'].mean():.1f}")
    print(f"        FRU inventory total: {total_fru}")


def main() -> int:
    build_borough_sector_csv()
    print()
    dissolve_sector_polygons()
    print()
    regenerate_l2_facilities()
    print()
    aggregate_fleet()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
