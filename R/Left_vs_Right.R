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

load_and_average <- function(filename) {
  
  data_raw <- read_delim(
    filename,
    delim = "\t",
    col_names = FALSE
  )
  
  # --------------------
  # Load, clean and format headers
  # --------------------
  
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
  
  
  ########## START OF CHATGPT CODE
  
  # convert to dataframe
  data_clean <- as.data.frame(data_clean)
  
  # metadata rows
  trial_row   <- as.character(unlist(data_clean[1, ]))
  subject_row <- as.character(unlist(data_clean[2, ]))
  joint_row   <- as.character(unlist(data_clean[3, ]))
  group_row   <- as.character(unlist(data_clean[4, ]))
  plane_row   <- as.character(unlist(data_clean[5, ]))
  
  # create grouping key
  avg_key <- paste(subject_row, joint_row, group_row, plane_row, sep = "_")
  unique_keys <- unique(avg_key[-1])  # unique groups, exclude first index column
  
  # initialize output with index column
  averaged_data <- data.frame(Index = data_clean[-(1:5), 1])
  
  # store metadata rows
  meta_trial   <- c(NA)
  meta_subject <- c(NA)
  meta_joint   <- c(NA)
  meta_group   <- c(NA)
  meta_plane   <- c(NA)
  
  # loop through each unique combination
  for (key in unique_keys) {
    
    cols <- which(avg_key == key)
    
    # average waveform columns
    avg_vals <- rowMeans(sapply(data_clean[-(1:5), cols], as.numeric), na.rm = TRUE)
    
    averaged_data[[key]] <- avg_vals
    
    # split metadata back out
    split_key <- str_split_fixed(key, "_", 4)
    
    meta_trial   <- c(meta_trial, 1)
    meta_subject <- c(meta_subject, split_key[1])
    meta_joint   <- c(meta_joint, split_key[2])
    meta_group   <- c(meta_group, split_key[3])
    meta_plane   <- c(meta_plane, split_key[4])
  }
  
  # prepend metadata rows
  averaged_data <- rbind(meta_trial, meta_subject, meta_joint, meta_group, meta_plane, averaged_data)
  rownames(averaged_data)[1:5] <- c("Trial","Subject","Joint","Group","Plane")
  
  return(averaged_data)
}

right_data <- load_and_average("H:/MultiCam/2025-10-07-reboot/04_24_2026/Width_1/Right_Leg.txt")

left_data <- load_and_average("H:/MultiCam/2025-10-07-reboot/11_28_2025/Width_1/Left_Leg.txt")

PLANES <- c("X", "Y", "Z")

N_TIME    <- 101
START_ROW <- 6

# --------------------
# Right vs Left RMSD comparison
# --------------------

JOINT_PAIRS <- data.frame(
  right = c("RPV", "RHIP", "RKNEE", "RANK"),
  left  = c("LPV", "LHIP", "LKNEE", "LANK"),
  stringsAsFactors = FALSE
)

side_results <- list()
counter <- 1

for (j in 1:nrow(JOINT_PAIRS)) {
  
  right_joint <- JOINT_PAIRS$right[j]
  left_joint  <- JOINT_PAIRS$left[j]
  
  for (plane_name in PLANES) {
    
    # ---------- Extract right side ----------
    mask_right <- as.vector(
      right_data[5, ] == plane_name &
        right_data[4, ] %in% c("G0","G1","G2","G3","G4") &
        right_data[3, ] == right_joint
    )
    
    data_right <- right_data[, mask_right]
    
    # ---------- Extract left side ----------
    mask_left <- as.vector(
      left_data[5, ] == plane_name &
        left_data[4, ] %in% c("G0","G1","G2","G3","G4") &
        left_data[3, ] == left_joint
    )
    
    data_left <- left_data[, mask_left]
    
    # ---------- Loop over subjects ----------
    subjects <- unique(as.character(unlist(data_right[2, ])))
    
    for (subj in subjects) {
      
      # ---------- Loop over camera groups ----------
      for (grp in c("G1","G2","G3","G4")) {
        
        # ----- Right side -----
        cols_g0_r <- which(
          data_right[2, ] == subj &
            data_right[4, ] == "G0"
        )
        
        cols_grp_r <- which(
          data_right[2, ] == subj &
            data_right[4, ] == grp
        )
        
        # ----- Left side -----
        cols_g0_l <- which(
          data_left[2, ] == subj &
            data_left[4, ] == "G0"
        )
        
        cols_grp_l <- which(
          data_left[2, ] == subj &
            data_left[4, ] == grp
        )
        
        # Skip if any waveform is missing
        if (length(cols_g0_r) == 0 ||
            length(cols_grp_r) == 0 ||
            length(cols_g0_l) == 0 ||
            length(cols_grp_l) == 0) next
        
        # Extract complete waveforms
        g0_right  <- as.numeric(unlist(data_right[START_ROW:(START_ROW + N_TIME - 1), cols_g0_r]))
        grp_right <- as.numeric(unlist(data_right[START_ROW:(START_ROW + N_TIME - 1), cols_grp_r]))
        
        g0_left   <- as.numeric(unlist(data_left[START_ROW:(START_ROW + N_TIME - 1), cols_g0_l]))
        grp_left  <- as.numeric(unlist(data_left[START_ROW:(START_ROW + N_TIME - 1), cols_grp_l]))

        # Compute RMSD over the entire gait cycle
        rmsd_right <- sqrt(mean((grp_right - g0_right)^2, na.rm = TRUE))
        rmsd_left  <- sqrt(mean((grp_left  - g0_left )^2, na.rm = TRUE))
        
        side_results[[counter]] <- data.frame(
          subject = subj,
          joint   = right_joint,
          plane   = plane_name,
          group   = grp,
          right_rmsd = rmsd_right,
          left_rmsd  = rmsd_left
        )
        
        counter <- counter + 1
      }
    }
  }
}

side_results <- bind_rows(side_results)

side_test <- t.test(
  side_results$right_rmsd,
  side_results$left_rmsd,
  paired = TRUE
)

print(side_test)

side_results$diff <- side_results$right_rmsd - side_results$left_rmsd

mean_diff <- mean(side_results$diff)
sd_diff   <- sd(side_results$diff)

cat("Mean paired difference (Right - Left):",
    round(mean_diff, 3), "degrees\n")
cat("SD of paired difference:",
    round(sd_diff, 3), "degrees\n")


hist_plot <- ggplot(side_results, aes(x = diff)) +
  
  geom_histogram(
    bins = 25,
    color = "grey60",
    fill = "grey80",   # lighter than grey80
    linewidth = 0.3
  ) +
  
  # Zero line
  geom_vline(
    aes(xintercept = 0, color = "Zero", linetype = "Zero"),
    linewidth = 0.4
  ) +
  
  # Mean difference line
  geom_vline(
    aes(xintercept = mean_diff, color = "Mean Difference", linetype = "Mean Difference"),
    linewidth = 0.4
  ) +
  
  scale_color_manual(
    name = NULL,
    values = c(
      "Zero" = "black",
      "Mean Difference" = "red"
    )
  ) +
  
  scale_linetype_manual(
    name = NULL,
    values = c(
      "Zero" = "solid",
      "Mean Difference" = "solid"
    )
  ) +
  
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0), add = c(0, 3))
  ) +
  
  labs(
    x = "Paired Difference (°)",
    y = "Count"
  ) +
  
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 8),
    
    # 👇 legend inside top-left
    legend.position = c(0.02, 0.98),
    legend.justification = c(0, 1),
    
    legend.title = element_blank(),
    legend.text = element_text(size = 7),
    legend.key.size = unit(0.4, "lines"),
    legend.spacing.x = unit(0.2, "cm"),
    
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 8),
    
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )
  
hist_plot

ggsave(
  filename = "Plots/Right_vs_Left/Histogram.png",
  plot = hist_plot,
  width = 4,
  height = 3,
  dpi = 600
)


# side_results <- side_results %>%
#   mutate(
#     mean_rmsd = (right_rmsd + left_rmsd) / 2,
#     diff_rmsd = right_rmsd - left_rmsd
#   )
# 
# bias <- mean(side_results$diff_rmsd)
# 
# loa_upper <- bias + 1.96 * sd(side_results$diff_rmsd)
# loa_lower <- bias - 1.96 * sd(side_results$diff_rmsd)
# 
# ba_plot <- ggplot(
#   side_results,
#   aes(x = mean_rmsd, y = diff_rmsd)
# ) +
#   geom_point(
#     alpha = 0.6,
#     size = 1.5
#   ) +
#   geom_hline(
#     yintercept = bias,
#     linewidth = 0.6
#   ) +
#   geom_hline(
#     yintercept = loa_upper,
#     linetype = "dashed",
#     linewidth = 0.5
#   ) +
#   geom_hline(
#     yintercept = loa_lower,
#     linetype = "dashed",
#     linewidth = 0.5
#   ) +
#   labs(
#     title = "Bland–Altman Plot",
#     x = "Mean RMSD (°)",
#     y = "Right - Left RMSD (°)"
#   ) +
#   theme_bw() +
#   theme(
#     plot.title = element_text(hjust = 0.5, size = 8),
#     axis.title = element_text(size = 8),
#     axis.text = element_text(size = 8),
#     panel.grid.major = element_blank(),
#     panel.grid.minor = element_blank()
#   )
# 
# ba_plot
