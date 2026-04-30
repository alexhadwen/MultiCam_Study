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
data_raw <- read_delim("H:/MultiCam/2025-10-07-reboot/04_24_2026/Width_2/Right_Leg.txt", delim = '\t', col_names = FALSE)

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

# --------------------
# Linear mixed effect modelling and bootstrapping
# --------------------

PLANES <- c("X","Y","Z")

N_TIME   <- 101
START_ROW <- 6 # where the time series data starts

all_results <- list() # initializing a results list

integrated_results <- data.frame(
  plane = character(),
  icc_integrated = numeric(),
  sem_integrated = numeric(),
  stringsAsFactors = FALSE
)

# looping over the 3 planes
for (plane_name in PLANES) {
  
  cat("Processing plane:", plane_name, "\n") # printing progress to the console
  
  # ----- Filter this plane ----- #
  mask <- as.vector(
    data_clean[5, ] == plane_name &
      data_clean[4, ] %in% c("G1","G2","G3","G4") &
      data_clean[3, ] %in% c("RKNEE")
  )
  
  data <- data_clean[, mask] # applying the mask to the dataset
  # data <- data[-c(3,5), ]

  # Storage for values of length N_TIME
  icc_values  <- numeric(N_TIME)
  icc_lower   <- numeric(N_TIME)
  icc_upper   <- numeric(N_TIME)
  
  sem_values  <- numeric(N_TIME)
  sem_lower   <- numeric(N_TIME)
  sem_upper   <- numeric(N_TIME)
  
  # ----- Time loop ----- #
  for (i in 1:N_TIME) { # each iteration is one timepoint in the waveform
    
    row_index <- START_ROW + (i - 1) # starts at row 6 onwards
    
    # separating each row as a vector and as a text value, each row now looks like subject|camera|stride|value
    df <- data.frame(
      stride  = as.numeric(unlist(data[1, ])),
      subject = as.character(unlist(data[2, ])),
      camera  = as.character(unlist(data[4, ])),
      trial   = as.numeric(unlist(data[row_index, ])) # value at the specific timepoint
    )
    
    df$subject <- as.factor(as.numeric(factor(df$subject))) # S01 -> 1
    df$camera <- as.factor(as.numeric(factor(df$camera))) # G1 -> 1
    
    # ----- Fit model ----- #
    # Fitting a linear mixed effects model
    # Response variable is trial, fixed effect is camera
    # subject is a random effect, stride within subject is random effect
    # REML is for estimating variance components
    # ----- Fit model ----- #
    model_tp <- lmer(
      trial ~ camera + (1 | subject) + (1 | subject:stride),
      data = df, REML = TRUE)
    
    # ----- ICC/SEM function ----- #
    stats_fun <- function(fit) { # takes the lmer fit, and extracts ICC and SEM
      
      vc <- as.data.frame(VarCorr(fit)) # all variance components from lmer
      
      var_subject <- vc$vcov[vc$grp == "subject"] # subject variance
      var_stride  <- vc$vcov[vc$grp == "subject:stride"] # stride variance
      var_error   <- attr(VarCorr(fit), "sc")^2 # square to get residual variance
      
      k <- 4 # I assumed an average number of strides of 4, this isn't necessarily correct
      
      icc <- var_subject / (var_subject + (var_stride + var_error)/k) # calculating ICC
      sem <- sqrt((var_stride + var_error)/k) # calculating SEM
      
      c(ICC = icc, SEM = sem) # return two total values under 'ICC' and 'SEM'
    }
    
    results <- stats_fun(model_tp) # calling stats function
    
    # storing values in an array
    icc_values[i] <- results["ICC"]
    sem_values[i] <- results["SEM"]
    
    ## ----- Bootstrapping ----- #
    boot_results <- bootMer(
      model_tp, # lmer model
      FUN = stats_fun, # get ICC and SEM
      nsim = 100, # number of simulations
      type = "parametric", # preserves nesting and variance structure
      use.u = FALSE, # random effects are resimulated each time
      parallel = "multicore", # runs on multiple cores
      ncpus = 4 # number of cores
    )

    # ----- CI ----- #
    # 95% confidence intervals, removes NA if did not converge
    icc_ci <- quantile(boot_results$t[,1], probs = c(0.025, 0.975), na.rm = TRUE)
    sem_ci <- quantile(boot_results$t[,2], probs = c(0.025, 0.975), na.rm = TRUE)

    #icc_ci <- c(0, 0.5)
    #sem_ci <- c(0, 0.1)
    
    icc_lower[i] <- icc_ci[1] # stores lower
    icc_upper[i] <- icc_ci[2] # stores higher
    sem_lower[i] <- sem_ci[1]
    sem_upper[i] <- sem_ci[2]
  }
  
  # ----- Integrated ICC ----- #
  icc_integrated <- mean(icc_values, na.rm = TRUE)
  sem_integrated <- mean(sem_values, na.rm = TRUE)
  
  integrated_results <- rbind(
    integrated_results,
    data.frame(
      plane = plane_name,
      icc_integrated = icc_integrated,
      sem_integrated = sem_integrated
    )
  )
  
  # ----- Store results ----- #
  all_results[[plane_name]] <- data.frame(
    x = seq(0, 100, length.out = N_TIME),
    plane = plane_name,
    icc = icc_values,
    icc_lower = icc_lower,
    icc_upper = icc_upper,
    sem = sem_values,
    sem_lower = sem_lower,
    sem_upper = sem_upper
  )
}

# ----- Save Integrated ICC Results ----- #

# write.table(integrated_results,
#             file = "Output_data/Integrated_Values/Right_Leg/w1_RKNEE_Integrated.txt",
#             row.names = FALSE,
#             col.names = TRUE,
#             sep = "\t")

# ----- ICC/SEM Plots ----- #

stats_x <- all_results[["X"]]
stats_y <- all_results[["Y"]]
stats_z <- all_results[["Z"]]

icc_x <- ggplot(stats_x, aes(x = x, y = icc)) +
  geom_ribbon(aes(ymin = icc_lower, ymax = icc_upper), fill = "orange", alpha = 0.25) +
  geom_line(color = "orange", linewidth = 0.6) +
  coord_cartesian(ylim = c(0,1)) +
  labs(x = NULL, y = "Pointwise ICC (3,k)") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme(axis.title.y = element_text(size = 8))


icc_y <- ggplot(stats_y, aes(x = x, y = icc)) +
  geom_ribbon(aes(ymin = icc_lower, ymax = icc_upper), fill = "orange", alpha = 0.25) +
  geom_line(color = "orange", linewidth = 0.6) +
  coord_cartesian(ylim = c(0,1)) +
  labs(x = NULL, y = NULL) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))


icc_z <- ggplot(stats_z, aes(x = x, y = icc)) +
  geom_ribbon(aes(ymin = icc_lower, ymax = icc_upper), fill = "orange", alpha = 0.25) +
  geom_line(color = "orange", linewidth = 0.6) +
  coord_cartesian(ylim = c(0,1)) +
  labs(x = NULL, y = NULL) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

sem_x <- ggplot(stats_x, aes(x = x, y = sem)) +
  geom_ribbon(aes(ymin = sem_lower, ymax = sem_upper), fill = "blue", alpha = 0.25) +
  geom_line(color = "blue", linewidth = 0.6) +
  labs(x = "Gait Cycle (%)", y = "Pointwise SEM (°)") +
  theme_bw() + 
  theme(axis.title.y = element_text(size = 8)) + 
  theme(axis.title.x = element_text(size = 8)) +
  ylim(0,2)

sem_y <- ggplot(stats_y, aes(x = x, y = sem)) +
  geom_ribbon(aes(ymin = sem_lower, ymax = sem_upper), fill = "blue", alpha = 0.25) +
  geom_line(color = "blue", linewidth = 0.6) +
  labs(x = "Gait Cycle (%)", y = NULL) +
  theme_bw() + 
  theme(axis.title.x = element_text(size = 8)) +
  ylim(0,2)

sem_z <- ggplot(stats_z, aes(x = x, y = sem)) +
  geom_ribbon(aes(ymin = sem_lower, ymax = sem_upper), fill = "blue", alpha = 0.25) +
  geom_line(color = "blue", linewidth = 0.6) +
  labs(x = "Gait Cycle (%)", y = NULL) +
  theme_bw() + 
  theme(axis.title.x = element_text(size = 8)) +
  ylim(0,2)

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

time_vec <- seq(0, 100, length.out = N_TIME) # time array

# converting to long data
df_long <- data.frame(
  time = rep(time_vec, times = ncol(wave_data)), # repeats the full vector once per column
  value = as.vector(wave_data), # as a vector
  meta[rep(1:nrow(meta), each = nrow(wave_data)), ] # applied the metadata to each point
)

RKNEE_df <- df_long %>% filter(joint == "RKNEE", group != "G0") # filtering

df_summary <- RKNEE_df %>%
  group_by(time, joint, plane, group) %>% # grouping by timepoint, joint, plane and camera group
  summarise(mean_value = mean(value, na.rm = TRUE),
    sd_value   = sd(value, na.rm = TRUE), .groups = "drop") %>%
  mutate( # creating a single standard band around the mean
    sd_lower = mean_value - sd_value,
    sd_upper = mean_value + sd_value,
    group = recode(group,
                   "G1" = "0.85 m",
                   "G2" = "1.65 m",
                   "G3" = "2.45 m",
                   "G4" = "3.30 m")
  )

df_summary_x <- df_summary %>% filter(plane == "X")
df_summary_y <- df_summary %>% filter(plane == "Y")
df_summary_z <- df_summary %>% filter(plane == "Z")

# ----- Raw data plotting ----- #
raw_x <- ggplot(df_summary_x, aes(x = time, y = mean_value, color = group, fill = group)) +
  geom_ribbon(aes(ymin = sd_lower, ymax = sd_upper), alpha = 0.25, color = NA) +
  geom_line(linewidth = 0.6) +
  labs(title = "Saggital Plane (flex(+)/ext(-))", x = NULL, y = "Angle (°)") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 10),
    legend.background = element_rect(fill = "white", color = "black"),
    legend.position = c(0.83, 0.25),
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.3, "cm")) +
  theme(axis.title.y = element_text(size = 8))

raw_y <- ggplot(df_summary_y, aes(x = time, y = mean_value, color = group, fill = group)) +
  geom_ribbon(aes(ymin = sd_lower, ymax = sd_upper), alpha = 0.25, color = NA) +
  geom_line(linewidth = 0.6) +
  labs(title = "Frontal Plane (abd(+)/(add(-))", x = NULL, y = NULL) +
  theme_bw() +
  theme(legend.position = "none") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))

raw_z <- ggplot(df_summary_z, aes(x = time, y = mean_value, color = group, fill = group)) +
  geom_ribbon(aes(ymin = sd_lower, ymax = sd_upper), alpha = 0.25, color = NA) +
  geom_line(linewidth = 0.6) +
  labs(title = "Transverse Plane (int(+)/(ext(-))", x = NULL, y = NULL) +
  theme_bw() +
  theme(legend.position = "none") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))

# Finished plot
complete_plot <- (raw_x | raw_y | raw_z ) / (icc_x | icc_y | icc_z) / (sem_x | sem_y | sem_z)

# Save the plot
ggsave(filename = "Plots/Right_Leg/W2_RKNEE.png", plot = complete_plot, width = 8, height = 6, dpi = 600)

# --------------------#
end_time = Sys.time()
time_taken <- end_time - start_time
print(time_taken) # print the total time taken
