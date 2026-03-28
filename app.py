# app.py — P2-ETF-MAGAT-ENGINE Streamlit Dashboard (with debug prints)
# ... (keep everything else the same as your current app.py, but replace render_history with the one below)

def render_history(hist_df: pd.DataFrame, master: pd.DataFrame, debug: bool = False):
    if hist_df.empty:
        st.info("Signal history will appear after the first training run.")
        return

    # Ensure numeric columns exist
    for col in ["actual_return", "hit"]:
        if col not in hist_df.columns:
            hist_df[col] = None

    # --- DEBUG: Print master index and columns ---
    st.write("**DEBUG: Master index sample (last 5):**", master.index[-5:])
    st.write("**DEBUG: Master columns containing '_ret':**", [c for c in master.columns if '_ret' in c][:10])
    # --- END DEBUG ---

    # Compute actual_return and hit for rows where they are missing
    for idx, row in hist_df.iterrows():
        if pd.isna(row.get("actual_return")):
            try:
                date = pd.Timestamp(row["signal_date"])
                col = f"{row['pick']}_ret"
                # --- DEBUG: Show what we are looking up ---
                st.write(f"DEBUG: Looking up {col} on {date.date()}")
                if col in master.columns and date in master.index:
                    ret = master.loc[date, col]
                    st.write(f"DEBUG: Retrieved ret = {ret}")
                    if not pd.isna(ret):
                        hist_df.at[idx, "actual_return"] = float(ret)
                        hist_df.at[idx, "hit"] = "✓" if ret > 0 else "✗"
                    else:
                        hist_df.at[idx, "actual_return"] = np.nan
                        hist_df.at[idx, "hit"] = "—"
                else:
                    st.write(f"DEBUG: Column {col} not in master or date {date.date()} not in index")
                    hist_df.at[idx, "actual_return"] = np.nan
                    hist_df.at[idx, "hit"] = "—"
            except Exception as e:
                st.write(f"DEBUG: Exception {e}")
                hist_df.at[idx, "actual_return"] = np.nan
                hist_df.at[idx, "hit"] = "—"

    if debug:
        st.write("**Debug: History Lookup Results**")
        debug_df = hist_df[["signal_date", "pick", "actual_return", "hit"]].copy()
        st.dataframe(debug_df)

    # Prepare display (same as before)
    disp = hist_df.sort_values("signal_date", ascending=False).copy()
    col_map = {
        "signal_date":   "Date",
        "pick":          "Pick",
        "conviction":    "Conviction",
        "actual_return": "Actual Return",
        "hit":           "Hit",
    }
    cols = [c for c in col_map if c in disp.columns]
    disp = disp[cols].rename(columns=col_map)

    if "Conviction" in disp.columns:
        disp["Conviction"] = disp["Conviction"].apply(
            lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else "—"
        )
    if "Actual Return" in disp.columns:
        disp["Actual Return"] = disp["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) and not pd.isna(x) else "—"
        )

    if "Hit" in disp.columns:
        hits  = (disp["Hit"] == "✓").sum()
        total = disp["Hit"].isin(["✓", "✗"]).sum()
        hr    = hits / total if total > 0 else 0
        st.markdown(
            f"<div class='hit-line'>Hit rate: <b>{hr:.1%}</b>"
            f" &nbsp;({hits}/{total} signals)</div>",
            unsafe_allow_html=True,
        )

    st.dataframe(disp, use_container_width=True, hide_index=True)
