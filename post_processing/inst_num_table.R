suppressPackageStartupMessages({
  library(ggplot2)
  library(plyr)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(forcats)
  library(viridis)
  library(Hmisc)
  library(knitr)
  library(kableExtra)
})

# (1) Stats by instances and QPUs
sdf_ibmq <- read.csv("./run_logs/summaries/ibm-qpu_stats.csv")
sdf_ibms <- read.csv("./run_logs/summaries/ibm-sim_stats.csv")
sdf_dwave <- read.csv("./run_logs/summaries/dwave_stats.csv")
sdf_quera <- read.csv("./run_logs/summaries/quera_stats.csv")

sdf = rbind(sdf_ibmq, sdf_ibms, sdf_dwave, sdf_quera)

res = sdf %>% group_by(across(all_of(c("instance_type",
                               "logtype")))) %>%
  summarise_at(c("success_runs", "failed_runs"),sum)

widetable <- pivot_wider(res, names_from = instance_type,
                      values_from=c(success_runs, failed_runs)) %>%
  select(c("logtype",
           "success_runs_MWC", "failed_runs_MWC",
           "success_runs_TSP", "failed_runs_TSP",
           "success_runs_UDMIS", "failed_runs_UDMIS"))

widetable %>% knitr::kable("markdown")

widetable %>% knitr::kable("latex") %>% save_kable("./figures/runs_table.tex")

## instdf <- sdf %>% mutate(
##                     solved = ifelse(success_runs > 0, 1, 0),
##                     failed = ifelse(success_runs == 0, 1, 0)) %>%
##   group_by(across(all_of(c("logtype", "instance_type")))) %>%
##   summarize_at(c("solved", "failed"), sum)

## wide_inst <- pivot_wider(instdf, names_from = instance_type,
##                          values_from=c(solved, failed)) %>%
##   select(c("solved_TSP", "failed_TSP", "solved_MWC", "failed_MWC",
##            "solved_UDMIS", "failed_UDMIS"))

## wide_inst %>% knitr::kable("markdown")

## wide_inst %>% knitr::kable("latex") %>% save_kable("./figures/inst_table.tex")

###
# let's count the number of unique instances solved on all devices
#
df = pivot_wider(select(sdf, -failed_runs), names_from=logtype, values_from=success_runs,
                 id_cols = c("instance_id", "instance_type", "qubo_vars"),
                 values_fill = 0)

df <- df %>%
  mutate(
    all_QPU = ifelse(((`IBM-QPU` > 0) & (DWave > 0) & (QuEra > 0)),
                     1, 0),
    all_devices = ifelse(((all_QPU == 1) & (`IBM-SIM` > 0)), 1, 0))

dfl <- df %>% pivot_longer(cols = c("IBM-QPU", "IBM-SIM", "DWave", "QuEra", "all_QPU", "all_devices"),
                     names_to="device", values_to = "successes")

unique_inst <- dfl %>% mutate( solved = successes > 0) %>%
  group_by(across(all_of(c("device", "instance_type")))) %>%
  summarize_at("solved", sum) %>%
  pivot_wider(names_from=instance_type, values_from=solved) %>%
  select(device, TSP, MWC, UDMIS) %>%
  mutate( Total = TSP + MWC + UDMIS)

unique_inst %>% kable("markdown")
unique_inst %>% kable("latex") %>% save_kable("./figures/inst_table.tex")
###
sdf_ibmq <- read.csv("./run_logs/summaries/ibm-qpu_stats.csv")
df_ibm_qpu <- read.csv("./run_logs/summaries/ibm-qpu_summary.csv")
df_ibm_qpu$device <- df_ibm_qpu$backend_name
df_ibm_qpu <- dplyr::filter(df_ibm_qpu, success == "True"& (!is.na(obj_from_QPU_sol)))

df_ibm_sim <- read.csv("./run_logs/summaries/ibm-sim_summary.csv")
df_ibm_sim$device <- df_ibm_sim$backend_name
df_ibm_sim <- dplyr::filter(df_ibm_sim, success == "True"& (!is.na(obj_from_QPU_sol)))

df_inst <- select(df_ibm_qpu, all_of(
                          c("device", "logfile", "instance_id", "instance_type",
                            "sol_time", "obj_from_QPU_sol")))

df_inst <- rbind(df_inst, select(df_ibm_sim, all_of(
                          c("device", "logfile", "instance_id", "instance_type",
                            "sol_time", "obj_from_QPU_sol"))))

df_dwave <- read.csv("./run_logs/summaries/dwave_summary.csv", stringsAsFactors = FALSE)
df_dwave$noemb_time = with(df_dwave, ifelse(embedding_time == -1, sol_time,
                                            sol_time - embedding_time))

edf = read.csv("./run_logs/summaries/embeddings.csv", stringsAsFactors = TRUE)
df_dwave = merge(x=df_dwave, y = select(edf, c("instance_id", emb_time, emb_success)), by = "instance_id")
df_dwave$sol_time = with(df_dwave, noemb_time + emb_time)

df_dwave <- filter(df_dwave, (success == "True") & (!is.na(obj_from_QPU_sol))) %>%
  dplyr::rename(c("device" = "chip_id",
         "logfile" = "filename"))

df_inst <- rbind(df_inst,
                 select(df_dwave, all_of(
                         c("device", "logfile", "instance_id", "instance_type",
                           "sol_time", "obj_from_QPU_sol"))))

df_quera <- read.csv("./run_logs/summaries/quera_summary.csv")
df_quera <- filter(df_quera, success == "True" & (!is.na(obj_from_QPU_sol)))
df_quera$device = rep("QuEra", length(df_quera$logfile))

df_inst <- rbind(df_inst,
                 select(df_quera, all_of(
                         c("device", "logfile", "instance_id", "instance_type",
                           "sol_time", "obj_from_QPU_sol"))))

str(df_inst)

dfc = read.csv("./run_logs/solutions.csv")
str(dfc)
dfc = plyr::rename(dfc, c("sol_time" = "classic_sol_time",
         "objective" = "classic_objective",
         "solution" = "classic_solution",
         "status" = "classic_status"))

str(dfc)
dfm = merge(x = df_inst, y = dfc, by = "instance_id")
dfm = plyr::rename(dfm, c("sol_time" = "QPU_soltime"))

dfm$device = ifelse(dfm$device %in% c("ibm_cusco","ibm_nazca"), "IBM QPU", dfm$device)
dfm$device = ifelse(dfm$device=="Advantage_system4.1", "DWave QPU", dfm$device)
dfm$device = ifelse(dfm$device=="ibmq_qasm_simulator", "IBM sim", dfm$device)
dfm$device = ifelse(dfm$device=="QuEra", "QuEra QPU", dfm$device)


ggplot(dfm) +
  geom_point(aes(x = log(QPU_soltime), y = log(classic_sol_time),
                 color=instance_type, shape=device), size=3, alpha=0.7) +
  geom_abline(slope = 1.0, intercept = 0.0, color = "red") +
  xlab("QPU solution time (total)") +
  ylab("Gurobi runtime, log sec") +
  theme(
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 13),
    axis.title.x = element_text(size = 26),
    axis.title.y = element_text(size = 26, margin = margin(t = 50)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    legend.position = c(0.85,0.8),
    legend.text = element_text(size=20),
    legend.title = element_text(size=20))+
  scale_colour_viridis_d()

ggsave("./figures/qpu_vs_cpu_runtimes.png", width = 16, height = 10)

dfm = filter(dfm, instance_id != "MWC1")  # that's a very degenerate case

dfm$QPU_rel_obj = with(dfm,
                       abs(obj_from_QPU_sol - classic_objective) / classic_objective)

ggplot(filter(dfm, device != "QuEra QPU")) +
  geom_histogram(aes(x = abs(QPU_rel_obj)), fill='blue',
                 position="dodge2",na.rm = TRUE)+
  xlab("Relative deviation between classical vs QPU objectives") +
  ylab("Instances count") +
  ggtitle("Objectives: (Solution from Gurobi) vs (Best of QPU results).")+
  theme(
    axis.text.x = element_text(size = 18, angle=90),
    axis.text.y = element_text(size = 13),
    axis.title.x = element_text(size = 26),
    axis.title.y = element_text(size = 26, margin = margin(t = 50)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = -90),
    strip.background = element_blank()
  )+
  scale_colour_viridis_d()+
  scale_fill_viridis_d()+
  facet_grid(device ~ instance_type, scales="free_y")+
  annotate("text", label = paste0("Within 90%: ", sum(dfm$QPU_rel_obj <= 0.1),
                                  " out of ", nrow(dfm), " (",
                                  sprintf("%0.1f%%",
                                          sum(dfm$QPU_rel_obj<=0.1)/nrow(dfm)*100),
                                  ")"), x =0.4, y=40, size=8)+
  annotate("text", label = paste0("Exactly 100%: ", sum(dfm$QPU_rel_obj == 0.0),
                                  " out of ", nrow(dfm), " (",
                                  sprintf("%0.1f%%",
                                          sum(dfm$QPU_rel_obj==0.0)/nrow(dfm)*100),
                                  ")"), x =0.4, y=30, size=8)

ggsave("./figures/obj_vs_classic.png", width = 16, height = 16)


ggplot(filter(dfm, device == "QuEra QPU")) +
  geom_histogram(aes(x = abs(QPU_rel_obj)), fill='blue')+
  xlab("Relative deviation between classical vs QPU objectives") +
  ylab("Instances count") +
  ggtitle("QuEra Objectives: (Solution from Gurobi) vs (Best of QPU results).")+
  theme(
    axis.text.x = element_text(size = 18, angle=90),
    axis.text.y = element_text(size = 13),
    axis.title.x = element_text(size = 26),
    axis.title.y = element_text(size = 26, margin = margin(t = 50)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = -90),
    strip.background = element_blank()
  )

ggsave("./figures/obj_vs_classic_QuEra.png", width=10, height=10)

df_summary = filter(dfm, !is.na(QPU_rel_obj)) %>% dplyr::group_by(instance_type, device) %>% dplyr::summarise(
    no_inst = n(),
    w50perc = sum(abs(.data[["QPU_rel_obj"]]) <= 0.5),
    w50perc_s = sum(abs(.data[["QPU_rel_obj"]]) <= 0.5) / n(),
    w20perc = sum(abs(.data[["QPU_rel_obj"]]) <= 0.2),
    w20perc_s = sum(abs(.data[["QPU_rel_obj"]]) <= 0.2) / n(),
    w1perc = sum(abs(.data[["QPU_rel_obj"]]) <= 0.01),
    w1perc_s = sum(abs(.data[["QPU_rel_obj"]]) <= 0.01) / n())

df_summary
latex(df_summary, file="./figures/qpu_vs_cpu.tex", digits=3)

# cross-check
with(filter(dfm, (device=="ibmq_qasm_simulator") & (instance_type=="TSP")),
    sum(QPU_rel_obj <= 0.5) / length(QPU_rel_obj))


# but the sizes are very different!

## dwavest = select(read.csv("./run_logs/summaries/dwave_stats.csv"),
##                  all_of(c("instance_id",  "qubo_vars")))
## ibmsst = select(read.csv("./run_logs/summaries/ibm-sim_stats.csv"),
##                  all_of(c("instance_id",  "qubo_vars")))
## ibmqst = select(read.csv("./run_logs/summaries/ibm-qpu_stats.csv"),
##                  all_of(c("instance_id",  "qubo_vars")))
## querast = select(read.csv("./run_logs/summaries/quera_stats.csv"),
##                  all_of(c("instance_id",  "qubo_vars")))

## sizesdf = dplyr::union(dwavest, ibmsst)
## sizesdf = dplyr::union(sizesdf, ibmqst)
## sizesdf = dplyr::union(sizesdf, querast)

## dfmm = merge(x = dfm, y = sizesdf, by = "instance_id")

## ggplot(dfmm) +
##   geom_histogram(aes(x = qubo_vars), fill='blue')+
##   xlab("No. of binary variables") +
##   ylab("Instances count") +
##   ggtitle("Instance sizes")+
##   theme(
##     axis.text.x = element_text(size = 18),
##     axis.text.y = element_text(size = 13),
##     axis.title.x = element_text(size = 26),
##     axis.title.y = element_text(size = 26, margin = margin(t = 50)),
##     panel.background = element_rect(fill = NA, color = "black"),
##     panel.grid.major = element_line(
##       size = 0.5, linetype = "solid",
##       color = "lightgrey"
##     ),
##     strip.text.x = element_text(size = 22),
##     strip.text.y = element_text(size = 22, angle = -90),
##     strip.background = element_blank()
##   )+
##   facet_grid(device ~ instance_type)

## ggsave("./figures/inst_sizes.png", width = 16, height = 10)

## with(dfmm,
##      table(device, instance_type))

## dfmm %>% group_by(instance_type, device) %>%
##   summarise(
##     no_inst = n())
