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
library(tidyr)

# --------------------
# Load, clean and format headers
# --------------------

# read a tab separated file with no headers
data_raw <- read_delim("H:/MultiCam/2025-10-07-reboot/04_24_2026/Width_1/Right_Leg.txt", delim = '\t', col_names = FALSE)

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

data_clean_mean <- averaged_data


JOINTS <- c("RPV", "RANK", "RKNEE", "RHIP")
PLANES <- c("X", "Y", "Z")

N_TIME    <- 101
START_ROW <- 6

all_results <- list()
counter <- 1

for (joint_name in JOINTS) {
  for (plane_name in PLANES) {
    
    cat("Processing:", joint_name, plane_name, "\n")
    
    mask <- as.vector(
      data_clean_mean[5, ] == plane_name &
        data_clean_mean[4, ] %in% c("G0","G1","G2","G3","G4") &
        data_clean_mean[3, ] == joint_name
    )
    
    data <- data_clean_mean[, mask]
    
    rmsd_df <- data.frame(
      x = seq(0, 100, length.out = N_TIME),
      G1 = NA_real_,
      G2 = NA_real_,
      G3 = NA_real_,
      G4 = NA_real_
    )
    
    # Loop through each point in the gait cycle
    for (i in 1:N_TIME) {
      
      row_index <- START_ROW + (i - 1)
      
      # Build dataframe for this time point
      df <- data.frame(
        subject = as.character(unlist(data[2, ])),
        camera  = as.character(unlist(data[4, ])),
        value   = as.numeric(unlist(data[row_index, ]))
      )
      
      # Wide format: one row per subject, one column per camera group
      df_wide <- reshape(
        df,
        idvar = "subject",
        timevar = "camera",
        direction = "wide"
      )
      
      # Calculate RMSD relative to G0
      for (grp in c("G1","G2","G3","G4")) {
        
        baseline <- df_wide$value.G0
        compare  <- df_wide[[paste0("value.", grp)]]
        
        # only use subjects with both values present
        valid <- complete.cases(baseline, compare)
        
        rmsd_df[i, grp] <- sqrt(
          mean((compare[valid] - baseline[valid])^2)
        )
      }
    }
    
    rmsd_df$joint <- joint_name
    rmsd_df$plane <- plane_name
    
    all_results[[counter]] <- rmsd_df
    counter <- counter + 1
  }
}

rmsd_results <- bind_rows(all_results)

# ----- Raw data plotting ----- #
group_colours <- c("0.85 m" = "#56B4E9",  # Blue
                   "1.65 m" = "#E69F00",  # Orange
                   "2.45 m" = "#009E73",  # Bluish Green
                   "3.30 m" = "#D55E00"   # Vermillion
)

group_scales <- list(
  scale_color_manual(values = group_colours),
  scale_fill_manual(values = group_colours)
)

plot_df <- rmsd_results %>%
  pivot_longer(
    cols = c(G1, G2, G3, G4),
    names_to = "group",
    values_to = "RMSD"
  ) %>%
  mutate(
    group = recode(
      group,
      "G1" = "0.85 m",
      "G2" = "1.65 m",
      "G3" = "2.45 m",
      "G4" = "3.30 m"
    ),
    group = factor(
      group,
      levels = c("0.85 m", "1.65 m", "2.45 m", "3.30 m")
    )
  )

make_rmsd_plot <- function(df, plot_title, show_legend = FALSE, x_label = NULL, y_label = NULL) {
  
  ggplot(
    df,
    aes(
      x = x,
      y = RMSD,
      colour = group,
      group = group
    )
  ) +
    geom_line(linewidth = 0.6) +
    scale_color_manual(
      name = "Camera Group",
      values = group_colours
    ) +
    scale_x_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, by = 20),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = c(0, 3),
      expand = c(0,0)
    ) +
    labs(
      title = plot_title,
      x = x_label,
      y = y_label
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 14),
      legend.key.size = unit(0.3, "cm"),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = if (show_legend) "bottom" else "none"
    )
}



# RANK
rank_x <- make_rmsd_plot(filter(plot_df, joint == "RANK", plane == "X"), "Sagittal Plane", y_label = "Ankle RMSD(°)")
rank_y <- make_rmsd_plot(filter(plot_df, joint == "RANK", plane == "Y"), "Frontal Plane")
rank_z <- make_rmsd_plot(filter(plot_df, joint == "RANK", plane == "Z"), "Transverse Plane")

# RKNEE
rknee_x <- make_rmsd_plot(filter(plot_df, joint == "RKNEE", plane == "X"), "", y_label = "Knee RMSD(°)")
rknee_y <- make_rmsd_plot(filter(plot_df, joint == "RKNEE", plane == "Y"), "")
rknee_z <- make_rmsd_plot(filter(plot_df, joint == "RKNEE", plane == "Z"), "")

# RHIP
rhip_x <- make_rmsd_plot(filter(plot_df, joint == "RHIP", plane == "X"), "", y_label = "Hip RMSD(°)")
rhip_y <- make_rmsd_plot(filter(plot_df, joint == "RHIP", plane == "Y"), "")
rhip_z <- make_rmsd_plot(filter(plot_df, joint == "RHIP", plane == "Z"), "")

# RPV
rpv_x <- make_rmsd_plot(filter(plot_df, joint == "RPV", plane == "X"), "", x_label = "Gait Cycle (%)", y_label = "Pelvis RMSD(°)")
rpv_y <- make_rmsd_plot(filter(plot_df, joint == "RPV", plane == "Y"), "", x_label = "Gait Cycle (%)")
rpv_z <- make_rmsd_plot(filter(plot_df, joint == "RPV", plane == "Z"), "", x_label = "Gait Cycle (%)")

rmsd_plot <-
  (rank_x  | rank_y  | rank_z)  /
  (rknee_x | rknee_y | rknee_z) /
  (rhip_x  | rhip_y  | rhip_z)  /
  (rpv_x   | rpv_y   | rpv_z) +
  plot_layout(guides = "collect") &
  theme(
    legend.position   = "bottom",
    legend.spacing.x  = unit(0.05, "cm"),
    legend.key.height = unit(0.3, "cm"),
    legend.key.width  = unit(0.6, "cm"),
    legend.box.just   = "bottom"
  )

rmsd_plot

ggsave(filename = "Plots/RMSD/Test.png", plot = rmsd_plot, width = 8, height = 6, dpi = 600)

### Summary values ###
rmsd_summary <- rmsd_results %>%
  pivot_longer(
    cols = c(G1, G2, G3, G4),
    names_to = "group",
    values_to = "RMSD"
  ) %>%
  mutate(
    group = recode(
      group,
      "G1" = "0.85 m",
      "G2" = "1.65 m",
      "G3" = "2.45 m",
      "G4" = "3.30 m"
    )
  ) %>%
  group_by(group) %>%
  summarise(
    Mean_RMSD = mean(RMSD, na.rm = TRUE),
    SD_RMSD   = sd(RMSD, na.rm = TRUE),
    .groups = "drop"
  )

rmsd_summary