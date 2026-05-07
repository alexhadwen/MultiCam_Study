rm(list = ls()) # clear the workspace
set.seed(123) # sets a fixed random seed
start_time <- Sys.time() # record the code start time
# -------------------- #

# ----- Libraries ----- #

library(dplyr) # data manipulation package
library(stringr) # string manipulation package
library(readr) # file reading and writing
library(lme4) # linear mixed effects models
library(ggplot2) # main plotting library
library(palmerpenguins) # used for examples and learning ggplot
library(ggthemes) # extra themes for ggplot2
library(patchwork) # combine multiple ggplots together

# --------------------
# Load, clean and format headers
# --------------------

# read a tab separated file with no headers
data_raw <- read_delim("H:/MultiCam/2025-10-07-reboot/04_24_2026/Width_3/Right_Leg.txt", delim = '\t', col_names = FALSE)

split_vals <- str_split_fixed(as.character(unlist(data_raw[4, ])), "_", 2) # splitting joint and group into two columns

joint_col <- split_vals[,1] # joint column
group_col <- split_vals[,2] # group column

colnames_joint <- colnames(data_raw) # initializing array
colnames_group  <- colnames(data_raw) # initializing array

joint_df <- as.data.frame(t(joint_col)) # transpose joint array into a row, makes it a dataframe
group_df <- as.data.frame(t(group_col)) # transpose group array into a row, makes it a dataframe

colnames(joint_df) <- colnames_joint # ensure column names match up
colnames(group_df)  <- colnames_group # ensure column names match up

# rebuilding the dataset, keeping row 1, adding joint and group rows, and keeping rows 5 onwards
data_clean <- bind_rows(data_raw[1, ], joint_df, group_df, data_raw[5:nrow(data_raw), ])

data_clean[1, ] <- t(str_replace(data_clean[1, ], ".*\\\\(S\\d+).*", "\\1")) # formats subject row
data_clean[3, ] <- t(str_replace(data_clean[3, ], ".*GROUP_(\\d+).*", "G\\1")) # formats group row

# --------------------
# Generate trial numbers
# --------------------

subject_row <- data_clean[1, ]
group_row   <- data_clean[3, ]
plane_row   <- data_clean[4, ]

trial_numbers <- numeric(ncol(data_clean)) # initializing a row with zeros for the trials numbers

last_key <- NULL # remembers previous group identity
counter <- 0 # counter for current trial number

for (col in 1:ncol(data_clean)) { # loops through the number of columns
  
  # the if statement is because the first column is just an index column
  if (col == 1) {
    trial_numbers[col] <- NA
    last_key <- NULL
    counter <- 0
  } else {
    
    key <- paste(subject_row[[col]], group_row[[col]], plane_row[[col]]) # the unique grouping
    
    # if the unique grouping is the same, add one to the trial, if different, start back at 1
    if (!identical(key, last_key)) {
      counter <- 1
    } else {
      counter <- counter + 1
    }
    
    trial_numbers[col] <- counter # store in the original array
    last_key <- key
  }
}

data_clean <- rbind(trial_numbers, data_clean) # prepends the trial numbers to the first row

# ----- Raw Data Plotting Setup ----- #
# changing this to column wise from row wise
meta <- data.frame(
  trial   = as.character(data_clean[1, -1]),
  subject = as.character(data_clean[2, -1]),
  joint   = as.character(data_clean[3, -1]),
  group   = as.character(data_clean[4, -1]),
  plane   = as.character(data_clean[5, -1]),
  stringsAsFactors = FALSE
)

wave_data <- data_clean[-c(1:5), -1] # extracting time series data
wave_data <- apply(wave_data, 2, as.numeric) # transposes the data

N_TIME   <- 101
time_vec <- seq(0, 100, length.out = N_TIME) # time array

# converting to long data
df_long <- data.frame(
  time = rep(time_vec, times = ncol(wave_data)), # repeats the full vector once per column
  value = as.vector(wave_data), # as a vector
  meta[rep(1:nrow(meta), each = nrow(wave_data)), ] # applied the metadata to each point
)

RANK_df <- df_long %>% filter(joint == "RANK", group != "G0") # filtering

df_summary <- RANK_df %>%
  group_by(time, joint, plane, subject, group) %>% # grouping by timepoint, joint, plane and camera group
  summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  mutate( # creating a single standard band around the mean
    group = recode(group,
                   "G1" = "0.85 m",
                   "G2" = "1.65 m",
                   "G3" = "2.45 m",
                   "G4" = "3.30 m")
  )

df_summary_x <- df_summary %>% filter(plane == "X")
df_summary_y <- df_summary %>% filter(plane == "Y")
df_summary_z <- df_summary %>% filter(plane == "Z")


library(rstatix)
df_summary_time <- df_summary_x[df_summary_x$time == 92, ]


df_summary_time %>% group_by(group) %>% shapiro_test(mean_value)

res.aov <- anova_test(data = df_summary_time, dv = mean_value, wid = subject, within = group)
results_anova <- get_anova_table(res.aov)

results_posthoc <- df_summary_x_92 %>% pairwise_t_test(mean_value ~ group, paired = TRUE, p.adjust.method = "bonferroni")


p_vals <- numeric(101)  # store 0–100
sig <- logical(101)

for (t in 0:100) {
  
  df_t <- df_summary_x[df_summary_x$time == t, ]
  
  res.aov <- anova_test(data = df_t, dv = mean_value, wid = subject, within = group)
  
  p_vals[t + 1] <- get_anova_table(res.aov)$p
  sig[t + 1] <- get_anova_table(res.aov)$p<.05
}

p_vals
sig
