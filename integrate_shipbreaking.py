import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Input files (converted in Step 1)
FILES = {
    2014: "Year2014.csv",
    2016: "Year2016.csv",
    2020: "Year2020.csv",
    2023: "Year2023.csv",
    2024: "Year2024.csv"
}

# Gold schema for with 10 columns as discussed in step 2
GOLD_SCHEMA = [
    "YEAR", "IMO", "NAME", "TYPE", "GT", "LDT",
    "BUILT", "LAST FLAG", "PLACE", "COUNTRY"
]

# Dictionary with all alias names to unify inconsistent column names across years
ALIAS = {
    "IMO": ["IMO", "IMO NUMBER", "IMO#", "IMO NO", "IMO NO."],
    "NAME": ["NAME", "NAME OF SHIP", "VESSEL", "VESSEL NAME"],
    "TYPE": ["TYPE", "TYPE OF SHIP", "SHIP TYPE", "TYPE OF VESSEL"],
    "GT": ["GT", "GROSS TONNAGE", "GROSS TONNAGE (GT)", "GROSS TONNAGE, GT"],
    "LDT": ["LDT", "LIGHT DISPLACEMENT TONNAGE", "LIGHTWEIGHT", "LIGHT WEIGHT"],
    "BUILT": ["BUILT", "BUILT IN (Y)", "YEAR BUILT", "BUILD YEAR"],
    "LAST FLAG": ["LAST FLAG", "FLAG", "CHANGE OF FLAG FOR BREAKING", "FLAG CHANGED FOR BREAKING"],
    "PLACE": ["PLACE", "DESTINATION CITY", "PLACE OF DEMOLITION", "LOCATION"],
    "COUNTRY": ["COUNTRY", "DESTINATION COUNTRY", "COUNTRY OF DEMOLITION"],
    "YEAR": ["YEAR"]
}

# First matching column name from the aliases list returned
def pick_col(df, candidates):
 
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# Standardizes one dataset to match the GOLD_SCHEMA:

def unify_one(path):

    df = pd.read_csv(path, dtype="unicode")
    df.columns = df.columns.astype(str).str.strip().str.upper()

    out = pd.DataFrame(index=df.index)
    for tgt in GOLD_SCHEMA:
        src = pick_col(df, ALIAS.get(tgt, []))
        out[tgt] = df[src] if src else pd.NA

    # Convert numeric fields
    for c in ["GT", "LDT", "BUILT", "YEAR"]:
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")

    # Clean string columns
    for c in ["IMO", "NAME", "TYPE", "LAST FLAG", "PLACE", "COUNTRY"]:
        out[c] = out[c].astype("string").str.strip()

    # check for invalid numeric values in GT, LDT and BUILT columns
    for c in ["GT", "LDT"]:
        out.loc[out[c].notna() & (out[c] <= 0), c] = np.nan

    out["BUILT"] = out["BUILT"].where(out["BUILT"].between(1900, 2035), pd.NA).astype("Int64")

    return out[GOLD_SCHEMA]


# Train a simple linear regression model: LDT ~ GT as discussed in the step 2 to impute missing LDT values
def train_ldt_model(df):
    
    train = df.dropna(subset=["GT", "LDT"])
    train = train[(train["GT"] > 0) & (train["LDT"] > 0)]
    if len(train) < 30:
        print(" Not enough valid data to train regression model.")
        return None

    X = train[["GT"]].to_numpy()
    y = train["LDT"].to_numpy()

    model = LinearRegression().fit(X, y)
    print(f"[MODEL] LDT = {model.coef_[0]:.3f} * GT + {model.intercept_:.2f} | R² = {model.score(X, y):.3f}")
    return model


def impute_missing_ldt(df, model):
    """Imputes missing LDT using regression model"""
    out = df.copy()

    if model is not None:
        req = out["LDT"].isna() & out["GT"].notna()
        if req.any():
            preds = np.maximum(model.predict(out.loc[req, ["GT"]]), 0)
            out.loc[req, "LDT"] = preds
            print(f" Regression-imputed LDT for {req.sum()} rows.")

    # Secondly - TYPE median applied  
    req = out["LDT"].isna() & out["TYPE"].notna()
    if req.any():
        medians = out.groupby("TYPE")["LDT"].median()
        out.loc[req, "LDT"] = out.loc[req, "TYPE"].map(medians)
        print(" median imputation applied.")

    # Lastly - overall median applied
    if out["LDT"].isna().any():
        median_val = out["LDT"].median()
        out["LDT"] = out["LDT"].fillna(median_val)
        print(f" Overall median ({median_val:.0f}) applied for remaining missing LDT values.")

    return out
    
##################### Main Function ###############################################################

def main():
    frames = []

    for yr, fname in FILES.items():
        path = Path(fname)
        if not path.exists():
            print(f" File not found: {fname}")
            continue

        df = unify_one(path)
        frames.append(df)
        print(f"- Loaded {fname} ({len(df)} rows)")

    if not frames:
        raise SystemExit("No files loaded. Check paths and filenames.")

    #  All year files combined
    comb_files = pd.concat(frames, ignore_index=True)

    # regression model for LDT ~ GT trained
    model = train_ldt_model(comb_files)

    # Impute missing LDT values
    comb_files = impute_missing_ldt(comb_files, model)

    # Remove invalid IMO rows
    before = len(comb_files)
    comb_files = comb_files.dropna(subset=["IMO"])
    print(f" Removed {before - len(comb_files)} rows with missing IMO.")

    # Drop duplicates and sort
    comb_files = comb_files.drop_duplicates(subset=["IMO", "NAME", "YEAR"], keep="first")
    comb_files = comb_files.sort_values(["YEAR", "COUNTRY", "PLACE", "TYPE", "NAME"]).reset_index(drop=True)

    # Save unified dataset
    out_path = "shipbreaking_unified.csv"
    comb_files.to_csv(out_path, index=False)
    print(f" Unified dataset saved: {out_path} ({len(comb_files)} rows)")


############## Execute #################################################################################

if __name__ == "__main__":
    main()
