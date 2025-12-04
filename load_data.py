#!/usr/bin/env python
# coding: utf-8

# # Loading Betting Data

# In[40]:


import os
import glob
import pandas as pd

# So printing dataframes doesn't explode the screen
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# If you're running this notebook from the project root:
BET_DATA_PATH = os.path.join("bet_data", "*.xls*")  # catches .xls and .xlsx


# In[41]:


files = sorted(glob.glob(BET_DATA_PATH))

print(f"Found {len(files)} tennis-data files:")
for f in files:
    print(os.path.basename(f))



# In[42]:


bet_dfs = []

for file in files:
    year = os.path.splitext(os.path.basename(file))[0]  # "2013" from "2013.xlsx"
    print(f"Loading {file} (year = {year})")
    
    # tennis-data.co.uk files are Excel, so use read_excel
    df = pd.read_excel(file)
    
    # Track source year/file just in case
    df["source_file"] = os.path.basename(file)
    df["source_year"] = int(year)
    
    bet_dfs.append(df)

bet_df_raw = pd.concat(bet_dfs, ignore_index=True)
print("Combined shape:", bet_df_raw.shape)
bet_df_raw.head()


# In[43]:


bet_df = bet_df_raw.copy()
bet_df.columns = bet_df.columns.str.strip().str.lower()
bet_df.columns


# In[44]:


bet_df["date"] = pd.to_datetime(bet_df["date"], errors="coerce")

# Quick sanity check
bet_df["date"].describe()


# In[45]:


# Keep only rows explicitly marked 'Completed' if that column exists
if "comment" in bet_df.columns:
    bet_df = bet_df[bet_df["comment"] == "Completed"]

# Also drop obvious walkovers in the score
if "score" in bet_df.columns:
    bet_df = bet_df[~bet_df["score"].astype(str).str.contains("W/O", case=False, na=False)]

print("After filtering completed matches:", bet_df.shape)


# In[46]:


for col in ["winner", "loser"]:
    if col in bet_df.columns:
        bet_df[col] = (
            bet_df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

bet_df[["winner", "loser"]].head()


# In[47]:


import numpy as np

# Implied probabilities from Bet365 odds (if present)
if {"b365w", "b365l"}.issubset(bet_df.columns):
    bet_df["b365w_prob"] = 1.0 / bet_df["b365w"]
    bet_df["b365l_prob"] = 1.0 / bet_df["b365l"]
    
    # Normalize so they sum roughly to 1 (remove overround)
    total_prob = bet_df["b365w_prob"] + bet_df["b365l_prob"]
    bet_df["b365w_prob_norm"] = bet_df["b365w_prob"] / total_prob
    bet_df["b365l_prob_norm"] = bet_df["b365l_prob"] / total_prob

# Rank difference (loser rank minus winner rank, so positive means winner was higher-ranked)
if {"wrank", "lrank"}.issubset(bet_df.columns):
    bet_df["rank_diff"] = bet_df["lrank"] - bet_df["wrank"]

bet_df[["winner", "loser", "wrank", "lrank", "rank_diff"]].head()


# In[48]:


print("Number of matches:", len(bet_df))
print("Unique winners:", bet_df["winner"].nunique())
print("Unique losers:", bet_df["loser"].nunique())

bet_df["surface"].value_counts()


# In[49]:


bet_df["b365w_prob_norm"].hist(bins=30)


# In[50]:


OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

bet_df.to_csv(os.path.join(OUTPUT_DIR, "bet_data_2010_2024_clean.csv"), index=False)
print("Saved cleaned betting data to processed/bet_data_2010_2024_clean.csv")


# # Loading ATP Data (Sackmann)

# In[51]:


import os
import glob
import pandas as pd

# Path to your Sackmann data
ATP_PATH = os.path.join("atp_match_data", "atp_matches_*.csv")

atp_files = sorted(glob.glob(ATP_PATH))

print(f"Found {len(atp_files)} Sackmann files:")
for f in atp_files:
    print(os.path.basename(f))


# In[52]:


atp_dfs = []

for file in atp_files:
    year = int(os.path.basename(file).split("_")[-1].split(".")[0])  # 2013 from atp_matches_2013.csv
    print(f"Loading {file} (year = {year})")
    
    df = pd.read_csv(file)
    df["source_year"] = year
    df["source_file"] = os.path.basename(file)
    
    atp_dfs.append(df)

atp_raw = pd.concat(atp_dfs, ignore_index=True)
print("Combined Sackmann shape:", atp_raw.shape)
atp_raw.head()


# In[53]:


atp_dfs = []

for file in atp_files:
    year = int(os.path.basename(file).split("_")[-1].split(".")[0])  # 2013 from atp_matches_2013.csv
    print(f"Loading {file} (year = {year})")
    
    df = pd.read_csv(file)
    df["source_year"] = year
    df["source_file"] = os.path.basename(file)
    
    atp_dfs.append(df)

atp_raw = pd.concat(atp_dfs, ignore_index=True)
print("Combined Sackmann shape:", atp_raw.shape)
atp_raw.head()


# In[54]:


atp = atp_raw.copy()

# Normalize column names
atp.columns = atp.columns.str.strip().str.lower()
atp.columns[:20]  # quick peek


# In[55]:


atp["tourney_date"] = pd.to_datetime(atp["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
atp["tourney_date"].describe()


# In[56]:


for col in ["winner_name", "loser_name"]:
    if col in atp.columns:
        atp[col] = (
            atp[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

atp[["winner_name", "loser_name"]].head()


# In[57]:


# Drop walkovers and unfinished matches based on score string
if "score" in atp.columns:
    mask_w_o = atp["score"].astype(str).str.contains("w/o", case=False, na=False)
    mask_ret = atp["score"].astype(str).str.contains("ret", case=False, na=False)
    mask_def = atp["score"].astype(str).str.contains("def", case=False, na=False) & ~atp["score"].str.contains("-", na=False)
    
    # Keep only matches that look like they have a normal score
    atp = atp[~(mask_w_o | mask_ret | mask_def)]

print("After removing walkovers/retirements:", atp.shape)


# In[58]:


atp = atp[(atp["source_year"] >= 2000) & (atp["source_year"] <= 2024)]
atp["source_year"].value_counts().sort_index()


# In[59]:


print("Rows:", len(atp))
print("Unique winners:", atp["winner_name"].nunique())
print("Unique losers:", atp["loser_name"].nunique())
atp["surface"].value_counts()
atp["tourney_level"].value_counts()  # G = slam, M = Masters, A = ATP, C = Challenger, F = Futures


# In[60]:


os.makedirs("processed", exist_ok=True)

atp.to_csv("processed/atp_matches_2000_2024_clean.csv", index=False)
print("Saved cleaned Sackmann data to processed/atp_matches_2000_2024_clean.csv")


# # Computing ELO System
# Jeff Sackmann Elo Formula 
# - K-factor dpeending on best of 3 vs best of 5
# - Surface specific adjustment
# - Aging decay factor
# - Margin of victory ignored
# - Draws never occur

# In[61]:


import pandas as pd
import numpy as np
from collections import defaultdict

# Copy the cleaned sackmann dataframe
elo_df = atp.copy()

# Sort all matches by date to process in correct chronological order
elo_df = elo_df.sort_values("tourney_date").reset_index(drop=True)

# Constants for Sackmann-style tennis Elo
BASE_ELO = 1500
K_BO3 = 32      # best-of-3 matches
K_BO5 = 48      # best-of-5 matches (higher because more reliable result)
DECAY = 0.999   # small rating decay per match to avoid rating inflation

# Initialize dictionaries
global_elo = defaultdict(lambda: BASE_ELO)
hard_elo   = defaultdict(lambda: BASE_ELO)
clay_elo   = defaultdict(lambda: BASE_ELO)
grass_elo  = defaultdict(lambda: BASE_ELO)

# To store outputs
pre_ELO_global_w = []
pre_ELO_global_l = []
pre_ELO_surface_w = []
pre_ELO_surface_l = []

# Map surface to the correct elo dict
surface_map = {
    "hard":  hard_elo,
    "clay":  clay_elo,
    "grass": grass_elo
}

def expected_score(elo_a, elo_b):
    """Expected probability that A beats B."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


for idx, row in elo_df.iterrows():
    w = row["winner_name"]
    l = row["loser_name"]
    surface = str(row["surface"]).lower()
    
    # If surface not recognized, default to global update only
    if surface not in surface_map:
        surface = None
    
    # Pre-match global Elo
    elo_w_global = global_elo[w]
    elo_l_global = global_elo[l]
    
    # Pre-match surface Elo
    if surface:
        elo_w_surface = surface_map[surface][w]
        elo_l_surface = surface_map[surface][l]
    else:
        elo_w_surface = np.nan
        elo_l_surface = np.nan
    
    # Store pre-match values
    pre_ELO_global_w.append(elo_w_global)
    pre_ELO_global_l.append(elo_l_global)
    pre_ELO_surface_w.append(elo_w_surface)
    pre_ELO_surface_l.append(elo_l_surface)
    
    # Determine K-factor
    k = K_BO5 if row["best_of"] == 5 else K_BO3
    
    # Expected scores (global)
    exp_w = expected_score(elo_w_global, elo_l_global)
    exp_l = 1 - exp_w
    
    # Update global Elo
    global_elo[w] = (elo_w_global * DECAY) + k * (1 - exp_w)
    global_elo[l] = (elo_l_global * DECAY) + k * (0 - exp_l)
    
    # Update surface-specific Elo
    if surface:
        exp_w_s = expected_score(elo_w_surface, elo_l_surface)
        exp_l_s = 1 - exp_w_s
        
        surface_map[surface][w] = (elo_w_surface * DECAY) + k * (1 - exp_w_s)
        surface_map[surface][l] = (elo_l_surface * DECAY) + k * (0 - exp_l_s)

# Attach results to dataframe
elo_df["elo_global_w_before"] = pre_ELO_global_w
elo_df["elo_global_l_before"] = pre_ELO_global_l
elo_df["elo_surface_w_before"] = pre_ELO_surface_w
elo_df["elo_surface_l_before"] = pre_ELO_surface_l

# Elo differences (winner minus loser)
elo_df["elo_global_diff"]  = elo_df["elo_global_w_before"]  - elo_df["elo_global_l_before"]
elo_df["elo_surface_diff"] = elo_df["elo_surface_w_before"] - elo_df["elo_surface_l_before"]

print("ELO calculation complete!")
elo_df.head()


# In[62]:


elo_df.to_csv("processed/atp_with_elo_2000_2024.csv", index=False)


# # Using H2H and recent form data

# In[63]:


elo_df   # <-- From Step 3


# In[64]:


import pandas as pd
from collections import defaultdict

df = elo_df.copy()

# Sort by date (already sorted in Step 3, but we ensure it again)
df = df.sort_values("tourney_date").reset_index(drop=True)

# Dictionaries to track past results
h2h_wins = defaultdict(lambda: defaultdict(int))

h2h_w_before = []
h2h_l_before = []
h2h_diff_list = []

for idx, row in df.iterrows():
    w = row["winner_name"]
    l = row["loser_name"]
    
    # Prior H2H values
    w_before = h2h_wins[w][l]
    l_before = h2h_wins[l][w]
    
    # Store pre-match H2H stats
    h2h_w_before.append(w_before)
    h2h_l_before.append(l_before)
    h2h_diff_list.append(w_before - l_before)
    
    # After match: update H2H
    h2h_wins[w][l] += 1

# Add to dataframe
df["h2h_w_before"] = h2h_w_before
df["h2h_l_before"] = h2h_l_before
df["h2h_diff"] = h2h_diff_list

print("H2H features computed!")
df[["winner_name","loser_name","h2h_w_before","h2h_l_before","h2h_diff"]].head(10)


# In[65]:


from collections import deque

# For storing past results
last5 = defaultdict(lambda: deque(maxlen=5))
last10 = defaultdict(lambda: deque(maxlen=10))

wins_last5 = []
wins_last10 = []
winpct_last10 = []

for idx, row in df.iterrows():
    w = row["winner_name"]
    l = row["loser_name"]
    
    # Pre-match stats
    wins_last5.append(len([x for x in last5[w] if x == 1]))
    wins_last10.append(len([x for x in last10[w] if x == 1]))
    
    if len(last10[w]) > 0:
        winpct_last10.append(sum(last10[w]) / len(last10[w]))
    else:
        winpct_last10.append(np.nan)
        
    # After match, update windows
    last5[w].append(1)
    last10[w].append(1)
    
    last5[l].append(0)
    last10[l].append(0)

# Add to df
df["wins_last5"] = wins_last5
df["wins_last10"] = wins_last10
df["winpct_last10"] = winpct_last10

print("Recent form features computed!")


# In[66]:


df.to_csv("processed/atp_with_elo_h2h_form_2000_2024.csv", index=False)
print("Saved enriched Sackmann dataset.")


# # Merging ATP and Betting Dataset

# In[67]:


import pandas as pd
import os

bet_path = os.path.join("processed", "bet_data_2010_2024_clean.csv")
atp_path = os.path.join("processed", "atp_with_elo_h2h_form_2000_2024.csv")

bet = pd.read_csv(bet_path, parse_dates=["date"])
atp_feat = pd.read_csv(atp_path, parse_dates=["tourney_date"])

print("Bet data shape:", bet.shape)
print("ATP w/ features shape:", atp_feat.shape)


# In[68]:


# Lowercase surface + round
for df_ in [bet, atp_feat]:
    if "surface" in df_.columns:
        df_["surface"] = df_["surface"].astype(str).str.strip().str.lower()
    if "round" in df_.columns:
        df_["round"] = df_["round"].astype(str).str.strip().str.upper()


# In[69]:


def full_to_short(name: str) -> str:
    """
    Convert 'roger federer' -> 'federer r'
    'juan martin del potro' -> 'del potro j' (approx)
    This is heuristic and not perfect, but works for most.
    """
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    if not name:
        return ""
    
    parts = name.split()
    if len(parts) == 1:
        # Single token: treat as last name only
        last = parts[0]
        first_initial = ""
    else:
        # Try to keep 'del potro', 'de minaur', 'van rijthoven' style prefixes
        # Simple heuristic: if second-to-last is a short preposition, join last two
        prefixes = {"de", "del", "van", "von", "da", "di", "la", "le"}
        if len(parts) >= 3 and parts[-2] in prefixes:
            last = " ".join(parts[-2:])      # 'del potro'
            first_initial = parts[0][0]      # first word initial
        else:
            last = parts[-1]
            first_initial = parts[0][0]
    
    return f"{last} {first_initial}".strip()


# In[70]:


def betname_to_short(name: str) -> str:
    """
    Convert 'gasquet r.' -> 'gasquet r'
    'del potro j.m.' -> 'del potro j'
    """
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    if not name:
        return ""
    
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    
    last = " ".join(parts[:-1])          # everything except final token
    first_token = parts[-1].replace(".", "")
    first_initial = first_token[0] if first_token else ""
    
    return f"{last} {first_initial}".strip()


# In[71]:


# In Sackmann
atp_feat["winner_short"] = atp_feat["winner_name"].apply(full_to_short)
atp_feat["loser_short"]  = atp_feat["loser_name"].apply(full_to_short)

# In bet data
bet["winner_short"] = bet["winner"].apply(betname_to_short)
bet["loser_short"]  = bet["loser"].apply(betname_to_short)

bet[["winner", "winner_short"]].head(10)


# In[72]:


# Choose which feature columns to merge from Sackmann
feat_cols = [
    "source_year", "surface", "round",
    "winner_short", "loser_short",
    "elo_global_w_before", "elo_global_l_before",
    "elo_global_diff",
    "elo_surface_w_before", "elo_surface_l_before",
    "elo_surface_diff",
    "h2h_w_before", "h2h_l_before", "h2h_diff",
    "wins_last5", "wins_last10", "winpct_last10"
]

# Ensure all exist
for c in feat_cols:
    if c not in atp_feat.columns:
        print("Missing column in atp_feat:", c)

atp_merge = atp_feat[feat_cols].copy()


# In[73]:


merged = bet.merge(
    atp_merge,
    how="left",
    left_on=["source_year", "surface", "round", "winner_short", "loser_short"],
    right_on=["source_year", "surface", "round", "winner_short", "loser_short"]
)

print("Merged shape:", merged.shape)
merged.head()


# In[74]:


merged["elo_global_diff"].isna().mean()


# # Robust Matching

# In[75]:


atp_feat = atp_feat.copy()
atp_feat["match_date"] = atp_feat["tourney_date"]


# In[76]:


# Pre-filter Sackmann rows where players match
pair_merged = bet.merge(
    atp_feat,
    how="left",
    left_on=["winner_short", "loser_short"],
    right_on=["winner_short", "loser_short"],
    suffixes=("_bet", "_atp")
)


# In[77]:


pair_merged["date_diff"] = (pair_merged["date_bet"] - pair_merged["match_date"]).abs().dt.days


# In[78]:


# bet: betting data
# atp_feat: Sackmann with Elo/H2H/form features

bet["date"] = pd.to_datetime(bet["date"], errors="coerce")
atp_feat["tourney_date"] = pd.to_datetime(atp_feat["tourney_date"], errors="coerce")

# Use tournament start date as proxy match date
atp_feat["match_date"] = atp_feat["tourney_date"]

# Ensure short name columns exist
# (you already computed these earlier)
# bet["winner_short"], bet["loser_short"]
# atp_feat["winner_short"], atp_feat["loser_short"]


# In[79]:


# Only keep the columns from atp_feat that we actually want to merge
feat_cols = [
    "source_year", "winner_short", "loser_short",
    "match_date",
    "elo_global_w_before", "elo_global_l_before",
    "elo_global_diff",
    "elo_surface_w_before", "elo_surface_l_before",
    "elo_surface_diff",
    "h2h_w_before", "h2h_l_before", "h2h_diff",
    "wins_last5", "wins_last10", "winpct_last10"
]

atp_merge = atp_feat[feat_cols].copy()

# Merge on year + players
pair_merged = bet.merge(
    atp_merge,
    how="left",
    on=["source_year", "winner_short", "loser_short"],
)


# In[80]:


pair_merged["date_diff"] = (pair_merged["date"] - pair_merged["match_date"]).abs().dt.days


# In[81]:


pair_merged = pair_merged[pair_merged["date_diff"] <= 3]


# In[82]:


pair_merged = pair_merged.sort_values(
    ["date", "winner_short", "loser_short", "date_diff"]
)

pair_merged = pair_merged.groupby(
    ["date", "winner_short", "loser_short"],
    as_index=False
).first()


# In[83]:


coverage = pair_merged["elo_global_diff"].notna().mean()
coverage


# In[84]:


pair_merged[["winner", "winner_short"]].head(20)



# In[85]:


len_bet = len(bet)
len_pair = len(pair_merged)
match_coverage = len_pair / len_bet
match_coverage


# In[86]:


pair_merged


# In[87]:


pair_merged.to_csv("processed/matches_2010_2024_with_features.csv", index=False)


# # EDA

# In[ ]:


matches = pair_merged.copy()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# If you haven't yet:
# matches = pair_merged.copy()

###########################################
# Plot 1: Histogram of implied win probabilities (Bet365)
###########################################

plt.figure(figsize=(8, 5))
matches["b365w_prob_norm"].dropna().hist(bins=30)
plt.xlabel("Normalized Bet365 implied probability (winner)")
plt.ylabel("Number of matches")
plt.title("Distribution of Implied Win Probabilities for Winners")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


###########################################
# Plot 2: Histogram of rank difference (lrank - wrank)
###########################################

plt.figure(figsize=(8, 5))
matches["rank_diff"].dropna().hist(bins=40)
plt.xlabel("Rank difference (loser rank - winner rank)")
plt.ylabel("Number of matches")
plt.title("Distribution of Rank Differences")
plt.axvline(0, linestyle="--")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


###########################################
# Plot 3: Elo difference vs implied probability (scatter)
###########################################

plt.figure(figsize=(8, 5))
subset = matches.dropna(subset=["elo_global_diff", "b365w_prob_norm"])

plt.scatter(
    subset["elo_global_diff"],
    subset["b365w_prob_norm"],
    alpha=0.2,
    s=10
)
plt.xlabel("Global Elo difference (winner - loser)")
plt.ylabel("Normalized Bet365 implied probability (winner)")
plt.title("Relationship Between Elo Difference and Market-Implied Probability")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


###########################################
# Plot 4: Match counts by surface
###########################################

plt.figure(figsize=(6, 4))
matches["surface"].value_counts().plot(kind="bar")
plt.xlabel("Surface")
plt.ylabel("Number of matches")
plt.title("Match Counts by Surface (2010–2024, matched subset)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[ ]:




