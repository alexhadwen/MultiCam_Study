rm(list = ls())

library(dplyr)
library(readr)
library(purrr)
library(tidyr)

# ---- 1. Define paths ----
base_path <- "Output_Data/Pairwise_Values"

width_folders <- paste0("width_", 1:3)

files <- file.path(
  base_path,
  width_folders,
  "pairwise_rpv_x_45.txt"
)

# ---- 2. Process function ----
process_file <- function(file, width_label) {
  
  df <- read_delim(file, delim = "\t", show_col_types = FALSE)
  
  df %>%
    select(contrast, p.value) %>%
    mutate(Width = toupper(width_label))
}

# ---- 3. Combine all widths ----
all_data <- map2_dfr(files, width_folders, process_file)

# ---- 4. Pivot wider ----
final_table <- all_data %>%
  pivot_wider(
    names_from = Width,
    values_from = p.value,
    names_glue = "p_{Width}"
  ) %>%
  arrange(contrast)

# ---- 5. Save ----
write.table(
  final_table,
  file = file.path(base_path, "combined_pairwise_rpv_x_45.txt"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)