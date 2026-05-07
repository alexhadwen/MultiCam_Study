import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports for ICC / ANOVA calculations
import statsmodels.api as sm
from statsmodels.formula.api import ols

import spm1d
import itertools


HEADER = {
    "subject": 0,
    "joint": 1,
    "group": 2,
    "plane": 3
}

# --------------------
# Global constants
# --------------------
N_TIME = 101
START_ROW = 4

GROUPS = ["G1", "G2", "G3", "G4"]
JOINTS = ["LPV", "LHIP", "LKNEE", "LANK"]
PLANES = ["X", "Y", "Z"]

# --------------------
# Functions
# --------------------

def load_and_clean_sift_data(file_path, sep='\t', n_strides=4):
    """
    Load and clean MultiCam kinematic data.

    Steps:
    - Load raw data
    - Keep first N strides per Subject–Joint–Plane
    - Clean subject and group naming conventions

    Parameters
    ----------
    file_path : str, Full path to the raw .txt file
    sep : str, optional, Column separator (default is tab)
    n_strides : int, optional, Number of strides to keep per Subject–Joint–Plane (default = 4)

    Returns: data_clean : pandas.DataFrame cleaned dataframe ready for analysis
    """

    # --------------------
    # Load raw data
    # --------------------
    data_raw = pd.read_csv(file_path, sep=sep, header=None)

    # Split group header row
    split_vals = (data_raw.iloc[3].astype(str).str.split('_', n=1, expand=True))

    # Create new rows
    prefix_row = split_vals[0]
    group_row = split_vals[1]

    # Insert new rows into dataframe
    data_raw = pd.concat([data_raw.iloc[:3], prefix_row.to_frame().T, group_row.to_frame().T, data_raw.iloc[4:]],ignore_index=True)

    # --------------------
    # Select first N strides per subject
    # --------------------
    header_info = pd.DataFrame({
        'col_index': data_raw.columns,
        'Subject': data_raw.iloc[0],
        'Stride_Label': data_raw.iloc[1],
        'Joint': data_raw.iloc[3],
        'Group': data_raw.iloc[4],
        'Plane': data_raw.iloc[5]
    })

    # Filtered headers to only keep the specified number of rows per subject
    filtered_headers = (header_info.groupby(['Subject', 'Joint', 'Plane'], group_keys=False).head(n_strides))

    keep_cols = [0] + filtered_headers['col_index'].tolist()
    data_lim = data_raw.loc[:, keep_cols]

    # --------------------
    # Clean and format headers
    # --------------------
    
    # Drop stride frame and P2D rows
    data_clean = data_lim.drop(index=[1, 2]).reset_index(drop=True)

    # Clean subject names (e.g., path → S0#)
    data_clean.iloc[HEADER['subject']] = data_clean.iloc[HEADER['subject']].replace(
        to_replace=r".*\\(S\d+).*",
        value=r"\1",
        regex=True
    )

    # Clean group labels (GROUP_# → G#)
    data_clean.iloc[HEADER['group']] = data_clean.iloc[HEADER['group']].replace(
        to_replace=r".*GROUP_(\d+).*",
        value=r"G\1",
        regex=True
    )

    # Reset index
    data_clean = data_clean.reset_index(drop = True)

    return data_clean


def select_columns(data, joint, plane, groups):
    return data.columns[
        (data.iloc[HEADER["joint"]] == joint) &
        (data.iloc[HEADER["plane"]] == plane) &
        (data.iloc[HEADER["group"]].isin(groups))
    ]


def icc_and_sem(data_t):
    model = ols('trial ~ C(subject) + C(camera)', data=data_t).fit()
    anova = sm.stats.anova_lm(model, typ=2)

    ms = anova['sum_sq'] / anova['df']

    n_subjects = data_t['subject'].nunique()
    n_cameras  = data_t['camera'].nunique()

    icc = (
        (ms['C(subject)'] - ms['Residual']) /
        (ms['C(subject)'] + ms['Residual']
         + (n_cameras / n_subjects) * (ms['C(camera)'] - ms['Residual']))
    )

    sem = np.sqrt(
        (anova.loc['C(camera)', 'sum_sq'] + anova.loc['Residual', 'sum_sq']) /
        (anova.loc['C(camera)', 'df'] + anova.loc['Residual', 'df'])
    )

    return icc, sem


def extract_waveform_matrix(data, group, plane, joint, start_row):
    cols = data.columns[
        (data.iloc[HEADER["joint"]] == joint) &
        (data.iloc[HEADER["plane"]] == plane) &
        (data.iloc[HEADER["group"]] == group)
    ]

    return data.loc[start_row:, cols].astype(float).to_numpy()


def compute_icc_sem_waveforms(data, joint, plane, groups=GROUPS, start_row=START_ROW, n_time=N_TIME):
    cols = select_columns(data, joint, plane, groups)
    subset = data.loc[:, cols]

    icc = np.zeros(n_time)
    sem = np.zeros(n_time)

    for i, row in enumerate(range(start_row, start_row + n_time)):
        df = subset.loc[[0, 2, row]].T
        df.columns = ['subject', 'camera', 'trial']

        df['subject'] = df['subject'].str.extract(r'S(\d+)').astype(int)
        df['camera']  = df['camera'].str.extract(r'G(\d+)').astype(int)
        df['trial']   = df['trial'].astype(float)

        icc[i], sem[i] = icc_and_sem(df)

    return icc, sem


def plot_stacked_joint_figures(data, joints=JOINTS, planes=PLANES, groups=GROUPS, start_row=START_ROW, n_time=N_TIME):
    x = np.linspace(0, 100, n_time)

    for joint in joints:

        fig, axes = plt.subplots(
            nrows=3,
            ncols=len(planes),
            figsize=(15, 9),
            sharex='col'
        )

        fig.suptitle(joint, fontsize=14)

        for col, plane in enumerate(planes):

            # --------------------
            # Row 1 — Raw waveforms
            # --------------------
            ax = axes[0, col]
            for group in groups:
                mat = extract_waveform_matrix(
                    data, group, plane, joint, start_row
                )
                ax.plot(x, mat.mean(axis=1), linewidth=1, label=group)

            ax.set_title(f"{plane} plane")
            ax.set_ylabel("Angle (deg)")
            ax.grid(alpha=0.3)

            # --------------------
            # Row 2 & 3 — ICC / SEM
            # --------------------
            icc, sem = compute_icc_sem_waveforms(
                data, joint, plane, groups, start_row, n_time
            )

            axes[1, col].plot(x, icc, linewidth=2)
            axes[1, col].set_ylabel("ICC")
            axes[1, col].set_ylim(0, 1)
            axes[1, col].grid(alpha=0.3)

            axes[2, col].plot(x, sem, linewidth=2)
            axes[2, col].set_ylabel("SEM (deg)")
            axes[2, col].set_xlabel("% Gait Cycle")
            axes[2, col].grid(alpha=0.3)

        axes[0, 0].legend(frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def plot_icc_sem_waveforms(data, joints=JOINTS, planes=PLANES, groups=GROUPS, start_row=START_ROW, n_time=N_TIME):
    x = np.linspace(0, 100, n_time)

    for joint in joints:
        for plane in planes:

            cols = select_columns(data, joint, plane, groups)
            subset = data.loc[:, cols]

            icc = np.zeros(n_time)
            sem = np.zeros(n_time)

            for i, row in enumerate(range(start_row, start_row + n_time)):
                df = subset.loc[[0, 2, row]].T
                df.columns = ['subject', 'camera', 'trial']

                df['subject'] = df['subject'].str.extract(r'S(\d+)').astype(int)
                df['camera']  = df['camera'].str.extract(r'G(\d+)').astype(int)
                df['trial']   = df['trial'].astype(float)

                icc[i], sem[i] = icc_and_sem(df)

            for values, label in [(icc, "ICC"), (sem, "SEM")]:
                plt.figure(figsize=(10, 5))
                plt.plot(x, values, label=label)
                plt.title(f"{joint} — {plane} plane")
                plt.xlabel("Normalized Gait Cycle (%)")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
    

def plot_raw_waveforms(data, joints=JOINTS, planes=PLANES, groups=GROUPS, start_row=START_ROW):

    for joint in joints:
        for plane in planes:

            plt.figure(figsize=(10, 5))
            plt.title(f"{joint} — {plane} plane")

            for group in groups:
                mat = extract_waveform_matrix(data, group, plane, joint, start_row)
                plt.plot(mat.mean(axis=1), label=group)

            plt.xlabel("Normalized Gait Cycle (%)")
            plt.ylabel("Angle (degrees)")
            plt.xlim(0, 100)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()


def plot_absolute_error_waveforms(data, joints=JOINTS, planes=PLANES, groups=GROUPS, start_row=START_ROW, n_time=N_TIME):
    x = np.linspace(0, 100, n_time)
    group_pairs = list(itertools.combinations(groups, 2))

    for joint in joints:
        for plane in planes:

            # ---- Compute mean curves on the fly ----
            mean_curves = {}
            for group in groups:
                mat = extract_waveform_matrix(data, group, plane, joint, start_row)
                mean_curves[group] = mat.mean(axis=1)

            # ---- Plot absolute errors ----
            plt.figure(figsize=(10, 5))
            plt.title(f"Absolute Error — {joint} — {plane} plane")

            for (g1, g2) in group_pairs:
                abs_err = np.abs(mean_curves[g1] - mean_curves[g2])
                plt.plot(x, abs_err, label=f"{g1} vs {g2}", linewidth=1)

            plt.xlabel("Normalized Gait Cycle (%)")
            plt.ylabel("Absolute Error (deg)")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()


def plot_spm_rmanova(data, joints=JOINTS, planes=PLANES, groups=GROUPS, start_row=START_ROW, alpha=0.05):
    for joint in joints:
        for plane in planes:

            # --------------------
            # Select columns
            # --------------------
            cols = data.columns[
                (data.iloc[HEADER["joint"]] == joint) &
                (data.iloc[HEADER["plane"]] == plane)
            ]

            # --------------------
            # Build data matrix Y (subjects × time)
            # --------------------
            Y = data.loc[start_row:, cols].astype(float).to_numpy().T

            # --------------------
            # Map groups to integers
            # --------------------
            group_labels = data.loc[HEADER["group"], cols].to_numpy()
            group_map = {f"G{i}": i for i in range(5)}  # keeps G0..G4
            A = np.array([group_map[g] for g in group_labels])


            # --------------------
            # Map subjects to integers
            # --------------------
            subj_labels = data.loc[HEADER["subject"], cols].to_numpy()
            subj_unique = np.unique(subj_labels)
            subj_map = {s: i for i, s in enumerate(subj_unique)}
            SUBJ = np.array([subj_map[s] for s in subj_labels])

            # --------------------
            # Run SPM RM-ANOVA
            # --------------------
            F = spm1d.stats.anova1rm(Y, A, SUBJ, equal_var=True)
            Fi = F.inference(alpha=alpha)

            # --------------------
            # Plot
            # --------------------
            plt.figure(figsize=(8,4))
            Fi.plot()
            plt.title(f"SPM RM-ANOVA: {joint} – {plane} plane")
            plt.xlabel("% Gait Cycle")
            plt.tight_layout()
            plt.show()


# --------------------
# Body of code
# --------------------

data_clean = load_and_clean_sift_data(r"H:\MultiCam\2025-10-07-reboot\11_28_2025\Width_1\Left_Leg.txt")

#plot_raw_waveforms(data_clean)

#plot_absolute_error_waveforms(data_clean)

plot_spm_rmanova(data_clean)

#plot_icc_sem_waveforms(data_clean)

#plot_stacked_joint_figures(data_clean)