import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import csv

st.set_page_config(layout="wide")
st.title("Expedition Wind Analysis")

# === Upload CSV ===
uploaded_file = st.file_uploader("Upload Expedition routing CSV", type="csv")

if uploaded_file:
    # Read CSV robustly
    rows = []
    reader = csv.reader(io.StringIO(uploaded_file.getvalue().decode("utf-8")), delimiter=',', quotechar='"')
    for row in reader:
        rows.append(row)

    header = rows[0]
    data_rows = [row for row in rows[1:] if len(row) == len(header)]
    df = pd.DataFrame(data_rows, columns=header)

    # === Parse fields ===
    df["Twa"] = pd.to_numeric(df.get("Twa", np.nan), errors="coerce")
    df["Tws"] = pd.to_numeric(df.get("Tws", np.nan), errors="coerce")

    # Detect TWD column
    twd_col_candidates = [col for col in df.columns if "twd" in col.lower()]
    if twd_col_candidates:
        twd_col = twd_col_candidates[0]
        df["Twd°M"] = pd.to_numeric(df[twd_col], errors="coerce")
    else:
        st.warning("No TWD column found in the CSV.")
        df["Twd°M"] = np.nan

    df = df.dropna(subset=["Twa", "Twd°M", "Tws"])

    # === Time info ===
    time_col = next((col for col in df.columns if "time" in col.lower()), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        start_time = df[time_col].min().strftime("%d-%b-%Y %H:%M")
        end_time = df[time_col].max().strftime("%d-%b-%Y %H:%M")
    else:
        start_time = end_time = "?"

    # === Model info from rows ===
    model_row = next((row for row in rows if any("model" in cell.lower() for cell in row if cell.strip())), None)
    model_name = ""
    if model_row:
        non_empty_cells = [cell.strip() for cell in model_row if cell.strip()]
        if len(non_empty_cells) > 1:
            model_name = non_empty_cells[1]

    # === Helper: plot TWD ===
    def plot_twd(df):
        dir_bins = np.arange(0, 361, 10)
        angles = np.deg2rad((dir_bins[:-1] + dir_bins[1:]) / 2)

        tws_bins = np.arange(0, 36, 4)
        tws_labels = [f"{tws_bins[i]}–{tws_bins[i+1]} kt" for i in range(len(tws_bins)-1)]

        df = df.copy()
        df["dir_bin"] = pd.cut(df["Twd°M"] % 360, bins=dir_bins, labels=False, include_lowest=True)
        df["tws_bin"] = pd.cut(df["Tws"], bins=tws_bins, labels=tws_labels, include_lowest=True)

        counts = df.groupby(["dir_bin", "tws_bin"]).size().unstack(fill_value=0)
        percentages = counts.reindex(index=np.arange(len(dir_bins)-1), fill_value=0)
        percentages = percentages / percentages.values.sum() * 100

        total_percent = percentages.sum(axis=1).values
        max_percent = np.ceil((total_percent + 1).max())
        ylim = int(np.ceil((max_percent + 2) / 2.0) * 2)
        rgrid_ticks = list(range(2, ylim + 1, 2))
        colors = ["#add8e6", "#9bddde", "#7fcdbb", "#66c2a5", "#90ee90", "#f0e68c", "#ffcccb", "#ffcc99"]

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 7))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        width = np.deg2rad(10)
        bottom = np.zeros(len(angles))
        for i, label in enumerate(tws_labels):
            heights = percentages[label].values if label in percentages.columns else np.zeros(len(angles))
            ax.bar(angles, heights, width=width, bottom=bottom,
                   color=colors[i % len(colors)], edgecolor='black', linewidth=0.5, label=label)
            bottom += heights

        bottom = np.zeros(len(angles))
        for i, label in enumerate(tws_labels):
            heights = percentages[label].values if label in percentages.columns else np.zeros(len(angles))
            for angle, h, b in zip(angles, heights, bottom):
                if h >= 1:
                    ax.text(angle, b + h / 2, f"{int(round(h))}%", ha='center', va='center', fontsize=8)
            bottom += heights

        for angle, total in zip(angles, total_percent):
            if total >= 3:
                ax.text(angle, total + 1, f"{int(round(total))}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
        ax.set_xticklabels([f"{d}°" for d in range(0, 360, 30)])
        ax.set_rgrids(rgrid_ticks, angle=90)
        ax.set_ylim(0, ylim)
        title_str = f"TWD vs TWS (% Time Sailed)\n{start_time} to {end_time}"
        if model_name:
            title_str += f"\n{model_name}"
        ax.set_title(title_str, va='bottom')
        ax.legend(title="TWS", loc="upper right", bbox_to_anchor=(1.2, 1.02))
        return fig

    # === Helper: plot TWA ===
    def plot_twa(df):
        bins = np.linspace(-180, 180, 37)
        angles = np.deg2rad((bins[:-1] + bins[1:]) / 2)

        tws_bins = np.arange(0, 36, 4)
        tws_labels = [f"{tws_bins[i]}–{tws_bins[i+1]} kt" for i in range(len(tws_bins)-1)]

        df = df.copy()
        df["twa_bin"] = pd.cut(df["Twa"], bins=bins, labels=False, include_lowest=True)
        df["tws_bin"] = pd.cut(df["Tws"], bins=tws_bins, labels=tws_labels, include_lowest=True)

        counts = df.groupby(["twa_bin", "tws_bin"]).size().unstack(fill_value=0)
        percentages = counts.reindex(index=np.arange(len(bins)-1), fill_value=0)
        percentages = percentages / percentages.values.sum() * 100

        total_percent = percentages.sum(axis=1).values
        max_percent = np.ceil((total_percent + 1).max())
        ylim = int(np.ceil((max_percent + 2) / 2.0) * 2)
        rgrid_ticks = list(range(2, ylim + 1, 2))
        colors = ["#add8e6", "#9bddde", "#7fcdbb", "#66c2a5", "#90ee90", "#f0e68c", "#ffcccb", "#ffcc99"]

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 7))
        ax.set_theta_zero_location("S")
        ax.set_theta_direction(-1)

        width = np.deg2rad(10)
        bottom = np.zeros(len(angles))
        for i, label in enumerate(tws_labels):
            heights = percentages[label].values if label in percentages.columns else np.zeros(len(angles))
            ax.bar(angles, heights, width=width, bottom=bottom,
                   color=colors[i % len(colors)], edgecolor='black', linewidth=0.5, label=label)
            bottom += heights

        bottom = np.zeros(len(angles))
        for i, label in enumerate(tws_labels):
            heights = percentages[label].values if label in percentages.columns else np.zeros(len(angles))
            for angle, h, b in zip(angles, heights, bottom):
                if h >= 1:
                    ax.text(angle, b + h / 2, f"{int(round(h))}%", ha='center', va='center', fontsize=8)
            bottom += heights

        for angle, total in zip(angles, total_percent):
            if total >= 3:
                ax.text(angle, total + 1, f"{int(round(total))}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

        tick_angles = np.arange(0, 361, 30)
        tick_labels = [f"{int(x - 180)}°" for x in tick_angles]
        ax.set_xticks(np.deg2rad(tick_angles))
        ax.set_xticklabels(tick_labels)
        ax.set_rgrids(rgrid_ticks, angle=90)
        ax.set_ylim(0, ylim)
        ax.set_title(f"TWA vs TWS (% Time Sailed)\n{start_time} to {end_time}", va='bottom')
        ax.legend(title="TWS", loc="upper right", bbox_to_anchor=(1.2, 1.02))
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("TWD vs TWS")
        st.pyplot(plot_twd(df))

    with col2:
        st.subheader("TWA vs TWS")
        st.pyplot(plot_twa(df))

    # === Time Series Plot ===
    if time_col:
        fig, ax1 = plt.subplots(figsize=(12, 9))

        ax1.plot(df[time_col], df["Tws"], color='blue', label='TWS')
        ax1.set_ylabel("TWS (kt)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        for x, y in zip(df[time_col], df["Tws"]):
            ax1.text(x, y, f"{int(round(y))}", color='blue', fontsize=7, va='bottom')

        ax2 = ax1.twinx()
        ax2.plot(df[time_col], df["Twd°M"], color='red', label='TWD')
        ax2.set_ylabel("TWD (°)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        for x, y in zip(df[time_col], df["Twd°M"]):
            ax2.text(x, y, f"{int(round(y))}", color='red', fontsize=7, va='top')

        # Plot vertical lines for marks if present
        mark_cols = [col for col in df.columns if "mark" in col.lower()]
        if mark_cols:
            mark_col = mark_cols[0]
            last_mark = None
            for t, m in zip(df[time_col], df[mark_col]):
                if pd.notna(m) and m != last_mark:
                    ax1.axvline(t, color='gray', linestyle='--', alpha=0.5)
                    ax1.text(t, ax1.get_ylim()[1], str(m), rotation=90, va='top', ha='right', fontsize=8)
                    last_mark = m

        plt.title("TWS/TWD Time Series")
        st.pyplot(fig)
else:
    st.info("Upload a CSV file to begin.")
