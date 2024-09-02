suppressPackageStartupMessages({
  library(ggplot2)
  library(plyr)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(forcats)
  library(viridis)
  library(Hmisc)
  library(ggh4x)
})

df_quera <- read.csv("./run_logs/summaries/quera_summary_full.csv")
df_quera$device = rep("QuEra", length(df_quera$logfile))
df_quera$inst_cat = with(df_quera,
                         ifelse(success=="True", ifelse(is.na(obj_from_QPU_sol),
                                                        "infeasible", "success"), "fail"))
df_quera$QPU = rep("NA-OPT", nrow(df_quera))

df_dwave <- read.csv("./run_logs/summaries/dwave_summary_full.csv", stringsAsFactors = FALSE)
df_dwave$noemb_time = with(df_dwave, ifelse(embedding_time == -1, sol_time,
                                            sol_time - embedding_time))

edf = read.csv("./run_logs/summaries/embeddings.csv", stringsAsFactors = TRUE)
df_dwave = merge(x=df_dwave, y = select(edf, c("instance_id", emb_time, emb_success)), by = "instance_id")
df_dwave$sol_time = with(df_dwave, noemb_time + emb_time)

df_dwave$inst_cat = with(df_dwave,
                         ifelse(success=="True", ifelse(is.na(obj_from_QPU_sol),
                                                        "infeasible", "success"), "fail"))

df_dwave <- df_dwave %>%
  dplyr::rename(c("device" = "chip_id",
         "logfile" = "filename"))

df_dwave$QPU = rep("QA-OPT", nrow(df_dwave))

df_ibm_qpu <- read.csv("./run_logs/summaries/ibm-qpu_summary_full.csv")
df_ibm_qpu$device <- df_ibm_qpu$backend_name

df_ibm_qpu$inst_cat = with(df_ibm_qpu,
                         ifelse(success=="True", ifelse(is.na(obj_from_QPU_sol),
                                                        "infeasible", "success"), "fail"))
df_ibm_qpu$QPU = rep("QAOA-OPT", nrow(df_ibm_qpu))

df_ibm_sim <- read.csv("./run_logs/summaries/ibm-sim_summary_full.csv")
df_ibm_sim$device <- df_ibm_sim$backend_name

df_ibm_sim$inst_cat = with(df_ibm_sim,
                         ifelse(success=="True", ifelse(is.na(obj_from_QPU_sol),
                                                        "infeasible", "success"), "fail"))
df_ibm_sim$QPU = rep("SIM-OPT", nrow(df_ibm_sim))

## Read the solutions from the classical baseline
dfc = read.csv("./run_logs/classic_solutions/solutions_all.csv")

str(dfc)
dfc = plyr::rename(dfc, c("sol_time" = "classic_sol_time",
         "objective" = "classic_objective",
         "solution" = "classic_solution",
         "status" = "classic_status"))

str(dfc)

df_inst <- select(df_quera, all_of(
                          c("QPU", "logfile", "instance_id", "instance_type", "qubo_vars",
                            "sol_time", "obj_from_QPU_sol", "inst_cat", "success")))

df_inst <- rbind(df_inst, select(df_ibm_qpu , all_of(
                          c("QPU", "logfile", "instance_id", "instance_type", "qubo_vars",
                            "sol_time", "obj_from_QPU_sol", "inst_cat", "success"))))

df_inst <- rbind(df_inst, select(df_ibm_sim , all_of(
                          c("QPU", "logfile", "instance_id", "instance_type", "qubo_vars",
                            "sol_time", "obj_from_QPU_sol", "inst_cat", "success"))))

df_inst <- rbind(df_inst,
                 select(df_dwave, all_of(
                         c("QPU", "logfile", "instance_id", "instance_type", "qubo_vars",
                           "sol_time", "obj_from_QPU_sol", "inst_cat", "success"))))

dfm = merge(x = df_inst, y = dfc, by = "instance_id")
dfm = plyr::rename(dfm, c("sol_time" = "QPU_soltime"))

dfm$prob_type = ifelse(dfm$instance_type=="MWC", "MaxCut",
                      ifelse(dfm$instance_type=="UDMIS", "UD-MIS", dfm$instance_type))

dfm$QPU_rel_obj = with(dfm,
                       abs(obj_from_QPU_sol - classic_objective) / classic_objective)

stats_by_instance = dfm %>%
  group_by(across(all_of(c("instance_id", "prob_type", "QPU", "qubo_vars")))) %>%
  summarise(
    feasible = sum(!is.na(obj_from_QPU_sol)),
    infeasible = sum(success == "True") - feasible,
    fail = sum(inst_cat == "fail"),
    total_runs = (feasible + infeasible + fail),
    cross_check = (total_runs == n()),
  ) %>% ungroup()

dfl = pivot_longer(stats_by_instance,
                   cols = c("feasible", "infeasible", "fail"),
                   names_to = "run_result",
                   values_to = "runs_count")

dfl$size_group = cut(dfl$qubo_vars,
                     breaks = c(4,25, 100,150),
                     labels = c("≤25", "26-100", "101+"))

dfl$QPU = factor(dfl$QPU, levels = c("NA-OPT", "QA-OPT", "QAOA-OPT", "SIM-OPT"))
dfl$prob_type = factor(dfl$prob_type, levels = c("UD-MIS", "MaxCut", "TSP"))

ggplot(dfl, aes(fill=factor(run_result,
                            levels = c("feasible", "infeasible", "fail")),
                x = factor(instance_id,
                           levels = unique(instance_id[order(qubo_vars, instance_id)]),
                           ordered=TRUE), y = runs_count))+
  geom_bar(stat="identity", position="stack", color='white', size=0.1)+
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_blank(), #element_text(size = 25),
        axis.title.y = element_blank(), #element_text(size = 25, margin = margin(t = 50)),
        panel.background = element_rect(fill = NA, color = "lightgray"),
        panel.grid.major = element_blank(), ## element_line(
        ##   size = 0.5, linetype = "solid",
        ##   color = "lightgrey"
        ##   ),
        strip.text.x = element_text(size = 12),
        strip.text.y = element_text(size = 20, angle = -90),
        strip.background = element_blank(), legend.position = c(0.8,0.875),
        legend.text = element_text(size=20),
        legend.title = element_text(size=20),
        ggh4x.facet.nestline = element_line(colour = "gray"))+
    facet_nested(QPU ~ prob_type + size_group, scales="free_x", space="free_x")+
    scale_fill_viridis_d("Run result", direction=-1)

ggsave("./figures/runs_summary.png", dpi=300, width = 15, height = 10)
