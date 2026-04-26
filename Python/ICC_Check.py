import pandas as pd
import numpy as np
from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
HEADER = {
    "subject": 0,
    "joint": 1,
    "group": 2,
    "plane": 3
}

N_TIME = 101
START_ROW = 3

# ============================================================
# LOAD + CLEAN FUNCTION
# ============================================================
def load_and_clean_sift_data(file_path, sep='\t', n_strides=4):

    data_raw = pd.read_csv(file_path, sep=sep, header=None)

    split_vals = data_raw.iloc[3].astype(str).str.split('_', n=1, expand=True)
    prefix_row = split_vals[0]
    group_row  = split_vals[1]

    data_raw = pd.concat(
        [data_raw.iloc[:3],
         prefix_row.to_frame().T,
         group_row.to_frame().T,
         data_raw.iloc[4:]],
        ignore_index=True
    )

    header_info = pd.DataFrame({
        'col_index': data_raw.columns,
        'Subject': data_raw.iloc[0],
        'Stride_Label': data_raw.iloc[1],
        'Joint': data_raw.iloc[3],
        'Group': data_raw.iloc[4],
        'Plane': data_raw.iloc[5]
    })

    filtered_headers = header_info.groupby(
        ['Subject', 'Joint', 'Plane'],
        group_keys=False
    ).head(n_strides)

    keep_cols = [0] + filtered_headers['col_index'].tolist()
    data_lim = data_raw.loc[:, keep_cols]

    data_clean = data_lim.drop(index=[1, 2]).reset_index(drop=True)

    data_clean.iloc[HEADER['subject']] = data_clean.iloc[HEADER['subject']].replace(
        to_replace=r".*\\(S\d+).*", value=r"\1", regex=True
    )

    data_clean.iloc[HEADER['group']] = data_clean.iloc[HEADER['group']].replace(
        to_replace=r".*GROUP_(\d+).*", value=r"G\1", regex=True
    )

    return data_clean


# ============================================================
# LOAD DATA
# ============================================================
data_clean = load_and_clean_sift_data(
    r"H:\MultiCam\2025-10-07-reboot\11_28_2025\Width_1\Left_Leg.txt"
)

subject_row = data_clean.iloc[0]
group_row   = data_clean.iloc[2]
plane_row   = data_clean.iloc[3]

trial_numbers = []

last_key = None
counter = 0

for col in data_clean.columns:
    if col == data_clean.columns[0]:  # index column
        trial_numbers.append(np.nan)
        last_key = None
        counter = 0
    else:
        key = (
            subject_row[col],
            group_row[col],
            plane_row[col]
        )

        if key != last_key:
            counter = 1
        else:
            counter += 1

        trial_numbers.append(counter)
        last_key = key

trial_row = pd.Series(trial_numbers, index=data_clean.columns)

data_clean = pd.concat(
    [trial_row.to_frame().T, data_clean],
    ignore_index=True
)

# ============================================================
# FILTER DATA (same as R)
# ============================================================

mask = (
    (data_clean.iloc[4] == "Y") &
    (data_clean.iloc[3].isin(["G1","G2","G3","G4"])) &
    (data_clean.iloc[2] == "LANK")
)

data = data_clean.loc[:, mask]

# Remove joint and plane rows (same as R: data[-c(3,5), ])
data = data.drop(index=[2,4]).reset_index(drop=True)

# ============================================================
# LOOP OVER TIME
# ============================================================

icc_values = np.zeros(N_TIME)

for i in range(N_TIME):

    row_index = START_ROW + i

    # ---- Build long dataframe for THIS timepoint ----
    df = pd.DataFrame({
        "stride":  pd.to_numeric(data.iloc[0].values),
        "subject": data.iloc[1].values,
        "camera":  data.iloc[2].values,
        "trial":   pd.to_numeric(data.iloc[row_index].values)
    })

    # ---- Convert to categorical factors ----
    df["subject"] = df["subject"].replace({
        "S01":1,"S02":2,"S03":3,"S04":4,
        "S05":5,"S06":6,"S07":7,"S08":8,"S09":9
    }).astype("category")

    df["camera"] = df["camera"].replace({
        "G1":1,"G2":2,"G3":3,"G4":4
    }).astype("category")

    df["stride"] = df["stride"].astype(int)

    # ---- Mixed Model (equivalent to lmer) ----
    model_tp = mixedlm(
        "trial ~ C(camera)",
        data=df,
        groups=df["subject"],
        re_formula="1",
        vc_formula={"subject_stride": "0 + C(subject):C(stride)"}
    )

    result = model_tp.fit(reml=True, disp=False)

    # ---- Extract variance components ----
    var_subject = result.cov_re.iloc[0, 0]
    var_stride  = result.vcomp[0]
    var_error   = result.scale

    k = 4

    icc_values[i] = (
        var_subject /
        (var_subject + (var_stride + var_error)/k)
    )
    
    #sem_values[i] = np.sqrt((var_stride + var_error)/k)


# ============================================================
# PLOT ICC
# ============================================================
x = np.linspace(0, 100, N_TIME)

plt.figure(figsize=(9,5))

plt.plot(x, icc_values, linewidth=2, label="ICC (MixedLM)")

plt.xlabel("% Gait Cycle")
plt.ylabel("ICC(3,k)")
plt.ylim([0,1])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
