suppressPackageStartupMessages({
  library(ggplot2)
  library(plyr)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(forcats)
  library(viridis)
  library(Hmisc)
})

df_quera <- read.csv("./run_logs/summaries/quera_summary.csv")
df_quera$device = rep("NA-OPT", length(df_quera$logfile))
df_quera$inst_cat = with(df_quera,
                         ifelse(success=="True", ifelse(is.na(obj_from_QPU_sol),
                                                        "infeasible", "success"), "fail"))
df_quera$QPU = rep("NA-OPT", nrow(df_quera))

df_dwave <- read.csv("./run_logs/summaries/dwave_summary.csv", stringsAsFactors = FALSE)
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

df_ibm_qpu <- read.csv("./run_logs/summaries/ibm-qpu_summary.csv")
df_ibm_qpu$device <- df_ibm_qpu$backend_name

df_ibm_qpu$inst_cat = with(df_ibm_qpu,
                         ifelse(success=="True", ifelse(is.na(obj_from_QPU_sol),
                                                        "infeasible", "success"), "fail"))
df_ibm_qpu$QPU = rep("QAOA-OPT", nrow(df_ibm_qpu))

df_ibm_sim <- read.csv("./run_logs/summaries/ibm-sim_summary.csv")
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
                            "sol_time", "obj_from_QPU_sol", "inst_cat")))

df_inst <- rbind(df_inst, select(df_ibm_qpu , all_of(
                          c("QPU", "logfile", "instance_id", "instance_type", "qubo_vars",
                            "sol_time", "obj_from_QPU_sol", "inst_cat"))))

df_inst <- rbind(df_inst, select(df_ibm_sim , all_of(
                          c("QPU", "logfile", "instance_id", "instance_type", "qubo_vars",
                            "sol_time", "obj_from_QPU_sol", "inst_cat"))))

df_inst <- rbind(df_inst,
                 select(df_dwave, all_of(
                         c("QPU", "logfile", "instance_id", "instance_type", "qubo_vars",
                           "sol_time", "obj_from_QPU_sol", "inst_cat"))))

dfm = merge(x = df_inst, y = dfc, by = "instance_id")
dfm = plyr::rename(dfm, c("sol_time" = "QPU_soltime"))

dfm$prob_type = ifelse(dfm$instance_type=="MWC", "MaxCut",
                      ifelse(dfm$instance_type=="UDMIS", "UD-MIS", dfm$instance_type))

dfm$QPU_rel_obj = with(dfm,
                       abs(obj_from_QPU_sol - classic_objective) / classic_objective)

ggplot(dfm)+
  geom_histogram(aes(x = qubo_vars, fill=inst_cat), position="stack")+
  geom_text(data = data.frame(
              qubo_vars = c(80,80),
              y=c(8,8),
              prob_type = c("TSP", "MaxCut"),
              QPU=c("NA-OPT", "NA-OPT"),
              label=c("(n/a)", "(n/a)")),
            mapping = aes(x = qubo_vars, y = y, label=label),
            size=10
            )+
  xlab("Number of binary variables") +
  ylab("Instances count") +
  theme(
    axis.text.x = element_text(size = 23, angle=90),
    axis.text.y = element_text(size = 23),
    axis.title.x = element_text(size = 26),
    axis.title.y = element_text(size = 26, margin = margin(t = 50)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = -90),
    strip.background = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(0.9, 0.125),
    legend.title = element_text(size=22),
    legend.text = element_text(size=22)
  )+
  scale_fill_viridis_d("Run result:")+
  facet_grid(factor(QPU, levels=c("NA-OPT", "QA-OPT", "QAOA-OPT",
                                  "SIM-OPT")) ~
                           factor(prob_type,
                                  levels=c("UD-MIS", "MaxCut", "TSP")))

ggsave("./figures/inst_summary.png", width = 15, height = 10)
