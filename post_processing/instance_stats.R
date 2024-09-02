# Generates stats with respect to *instances* (not devices)
suppressPackageStartupMessages({
  library(plyr)
  library(dplyr)
  library(tidyr)
})

dstats = read.csv("./run_logs/summaries/all_devices_stats_full.csv")

istats = pivot_wider(dstats, id_cols = c("instance_id", "instance_type", "qubo_vars"),
                     names_from = "logtype",
                     values_fill = 0,
                     values_from = c("success_runs", "failed_runs"))

write.csv(istats, "./run_logs/summaries/inst_sol_summary.csv",
          row.names = FALSE)
