import pandas as pd
from collections import defaultdict, deque
from pathlib import Path
import streamlit as st

# ---------- Config ----------
ATP_CLEAN = Path("processed/atp_matches_2000_2024_clean.csv")
ATP_URL = "https://raw.githubusercontent.com/LukeF2/tennis_betting_pred/main/processed/atp_matches_2000_2024_clean.csv"
BASE_ELO, K_BO3, K_BO5, DECAY = 1500, 32, 48, 0.999
prefixes = {"de","del","van","von","da","di","la","le"}

# ---------- Helpers ----------
def full_to_short(name: str) -> str:
    if not isinstance(name, str): return ""
    name = name.strip().lower()
    if not name: return ""
    parts = name.split()
    if len(parts) == 1:
        last, first = parts[0], ""
    elif len(parts) >= 3 and parts[-2] in prefixes:
        last, first = " ".join(parts[-2:]), parts[0][0]
    else:
        last, first = parts[-1], parts[0][0]
    return f"{last} {first}".strip()

def betname_to_short(name: str) -> str:
    if not isinstance(name, str): return ""
    name = name.strip().lower()
    if not name: return ""
    parts = name.split()
    if len(parts) == 1: return parts[0]
    last = " ".join(parts[:-1])
    first_token = parts[-1].replace(".", "")
    first = first_token[0] if first_token else ""
    return f"{last} {first}".strip()

def normalize_surface(s) -> str:
    s = str(s).strip().lower()
    if s in ("nan","none",""): return ""
    if "hard" in s: return "hard"
    if "clay" in s: return "clay"
    if "grass" in s: return "grass"
    return ""

def expected_score(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))

# ---------- Data load & state build ----------
@st.cache_resource(show_spinner=True)
def load_atp():
    if ATP_CLEAN.exists():
        return pd.read_csv(ATP_CLEAN)
    return pd.read_csv(ATP_URL)

@st.cache_resource(show_spinner=True)
def build_state():
    atp = load_atp()
    atp.columns = atp.columns.str.lower()
    atp["tourney_date"] = pd.to_datetime(atp["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    atp = atp.sort_values("tourney_date")

    global_elo = defaultdict(lambda: BASE_ELO)
    hard_elo = defaultdict(lambda: BASE_ELO)
    clay_elo = defaultdict(lambda: BASE_ELO)
    grass_elo = defaultdict(lambda: BASE_ELO)
    surf_map = {"hard": hard_elo, "clay": clay_elo, "grass": grass_elo}

    h2h = defaultdict(lambda: defaultdict(int))
    last5 = defaultdict(lambda: deque(maxlen=5))
    last10 = defaultdict(lambda: deque(maxlen=10))

    for _, row in atp.iterrows():
        w = full_to_short(row.get("winner_name", ""))
        l = full_to_short(row.get("loser_name", ""))
        surface = normalize_surface(row.get("surface"))
        k = K_BO5 if row.get("best_of", 3) == 5 else K_BO3

        wg, lg = global_elo[w], global_elo[l]
        ws = surf_map[surface][w] if surface in surf_map else None
        ls = surf_map[surface][l] if surface in surf_map else None

        exp_w = expected_score(wg, lg)
        global_elo[w] = (wg * DECAY) + k * (1 - exp_w)
        global_elo[l] = (lg * DECAY) + k * (0 - (1 - exp_w))

        if surface in surf_map:
            exp_ws = expected_score(ws, ls)
            surf_map[surface][w] = (ws * DECAY) + k * (1 - exp_ws)
            surf_map[surface][l] = (ls * DECAY) + k * (0 - (1 - exp_ws))

        h2h[w][l] += 1
        last5[w].append(1); last10[w].append(1)
        last5[l].append(0); last10[l].append(0)

    # return plain dicts (no lambdas)
    return {
        "global_elo": dict(global_elo),
        "hard_elo": dict(hard_elo),
        "clay_elo": dict(clay_elo),
        "grass_elo": dict(grass_elo),
        "h2h": {k: dict(v) for k, v in h2h.items()},
        "last5": {k: list(v) for k, v in last5.items()},
        "last10": {k: list(v) for k, v in last10.items()},
    }

state = build_state()

def get_state():
    ge = defaultdict(lambda: BASE_ELO, state["global_elo"])
    he = defaultdict(lambda: BASE_ELO, state["hard_elo"])
    ce = defaultdict(lambda: BASE_ELO, state["clay_elo"])
    gre = defaultdict(lambda: BASE_ELO, state["grass_elo"])
    surf_map = {"hard": he, "clay": ce, "grass": gre}

    h2h = defaultdict(lambda: defaultdict(int))
    for k, v in state["h2h"].items():
        h2h[k] = defaultdict(int, v)

    last5 = defaultdict(lambda: deque(maxlen=5))
    for k, v in state["last5"].items():
        last5[k].extend(v)
    last10 = defaultdict(lambda: deque(maxlen=10))
    for k, v in state["last10"].items():
        last10[k].extend(v)

    return ge, surf_map, h2h, last5, last10

# ---------- Core computation ----------
def compute_for_tournament(upload_df: pd.DataFrame, tournament_name: str) -> pd.DataFrame:
    GLOBAL_ELO, SURF_MAP, H2H, LAST5, LAST10 = get_state()

    df = upload_df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["tournament"].str.contains(tournament_name, case=False, na=False)].copy()
    df["surface_norm"] = df.get("surface", "").apply(normalize_surface)
    df["winner_short"] = df["winner"].apply(betname_to_short)
    df["loser_short"]  = df["loser"].apply(betname_to_short)

    # Bet365 implied probs
    df["b365w_prob"] = 1 / df["b365w"]
    df["b365l_prob"] = 1 / df["b365l"]
    total = df["b365w_prob"] + df["b365l_prob"]
    df["b365w_prob_norm"] = df["b365w_prob"] / total
    df["b365l_prob_norm"] = df["b365l_prob"] / total

    rows = []
    for _, r in df.iterrows():
        w, l = r["winner_short"], r["loser_short"]
        surf = r["surface_norm"]

        wg, lg = GLOBAL_ELO[w], GLOBAL_ELO[l]
        ws = SURF_MAP[surf][w] if surf in SURF_MAP else None
        ls = SURF_MAP[surf][l] if surf in SURF_MAP else None

        h2h_w, h2h_l = H2H[w][l], H2H[l][w]
        wins5 = len([x for x in LAST5[w] if x == 1])
        wins10 = len([x for x in LAST10[w] if x == 1])
        winpct10 = (sum(LAST10[w]) / len(LAST10[w])) if len(LAST10[w]) else None

        elo_diff = wg - lg
        elo_surf_diff = (ws - ls) if ws is not None and ls is not None else None
        prob = 1 / (1 + 10 ** (-elo_diff / 400))
        edge = prob - r["b365w_prob_norm"]

        rows.append({
            "date": r["date"], "tournament": r["tournament"],
            "winner": r["winner"], "loser": r["loser"],
            "b365w": r["b365w"], "b365l": r["b365l"],
            "b365w_prob_norm": r["b365w_prob_norm"],
            "elo_implied_prob": prob,
            "elo_minus_market": edge,
            "elo_global_diff": elo_diff,
            "elo_surface_diff": elo_surf_diff,
            "h2h_diff": h2h_w - h2h_l,
            "wins_last5": wins5, "wins_last10": wins10, "winpct_last10": winpct10,
        })

    out_df = pd.DataFrame(rows).drop_duplicates(subset=["date","winner","loser"])
    return out_df.sort_values("date")

# ---------- UI ----------
st.title("Tournament Implied Probabilities (Elo vs Bet365)")

uploaded = st.file_uploader("Upload matches file (CSV or Excel)", type=["csv","xlsx","xls"])
tournament = st.text_input("Tournament name", value="masters cup")

if uploaded and tournament:
    if uploaded.name.endswith(".csv"):
        input_df = pd.read_csv(uploaded)
    else:
        input_df = pd.read_excel(uploaded)
    results = compute_for_tournament(input_df, tournament)
    st.write(f"Rows: {len(results)}")
    st.dataframe(results)
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="tournament_probs.csv", mime="text/csv")
else:
    st.info("Upload a file and enter a tournament name to see results.")

