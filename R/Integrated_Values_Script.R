rm(list = ls()) # clear the workspace

library(dplyr)
library(readr)
library(purrr)
library(tidyr)

# ---- 1. Read files ----
files <- list.files(
  path = "C:/Users/Customer/Downloads/Local Queens/Grad School/HMRL_Local/CH_vs_VW/R/Integrated_Values",
  pattern = "*.txt",
  full.names = TRUE
)

# ---- 2. Function to process each file ----
process_file <- function(file) {
  
  df <- read_delim(file, delim = "\t", show_col_types = FALSE)
  fname <- basename(file)
  
  width <- sub("_.*", "", fname)  # w1, w2, w3
  joint <- sub("w[0-9]+_(.*?)_Integrated\\.txt", "\\1", fname)
  
  df %>%
    mutate(
      Joint = joint,
      Width = toupper(width)   # W1, W2, W3
    ) %>%
    select(Joint, Plane = plane, Width, ICC = icc_integrated, SEM = sem_integrated)
}

# ---- 3. Combine all files ----
all_data <- map_dfr(files, process_file)

# ---- 4. Pivot wider ----
final_table <- all_data %>%
  pivot_wider(
    names_from = Width,
    values_from = c(ICC, SEM),
    names_glue = "{.value}_{Width}"
  ) %>%
  arrange(Joint, Plane)

# ---- 5. Save ----
write.table(final_table,
            file = "Integrated_Values/Integrated_Values.txt",
            sep = "\t",
            row.names = FALSE,
            quote = FALSE)
