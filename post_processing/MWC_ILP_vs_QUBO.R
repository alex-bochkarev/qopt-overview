library(ggplot2)
library(viridis)
library(dplyr)
df <- read.csv("./run_logs/classic_solutions/MWC_QUBO_vs_LBOP/MWC.csv")

# cross-checks
# are the instances unique

max(table(df$instance_id))  # yes, they are

# Let's pick the instances that were (successfully) tested
# against our QPUs

df_ibm_qpu <- read.csv("./run_logs/summaries/ibm-qpu_summary.csv")
df_ibm_qpu <- dplyr::filter(df_ibm_qpu, success == "True"& (!is.na(obj_from_QPU_sol)))

df_ibm_sim <- read.csv("./run_logs/summaries/ibm-sim_summary.csv")
df_ibm_sim <- dplyr::filter(df_ibm_sim, success == "True"& (!is.na(obj_from_QPU_sol)))

df_inst <- select(df_ibm_qpu, all_of(
                          c("logfile", "instance_id", "instance_type",
                            "sol_time", "obj_from_QPU_sol")))

df_inst <- rbind(df_inst, select(df_ibm_sim, all_of(
                          c("logfile", "instance_id", "instance_type",
                            "sol_time", "obj_from_QPU_sol"))))

df_dwave <- read.csv("./run_logs/summaries/dwave_summary.csv", stringsAsFactors = FALSE)
df_dwave$noemb_time = with(df_dwave, ifelse(embedding_time == -1, sol_time,
                                            sol_time - embedding_time))

df_dwave <- filter(df_dwave, (success == "True") & (!is.na(obj_from_QPU_sol))) %>%
  dplyr::rename(c("device" = "chip_id",
         "logfile" = "filename"))

df_inst <- rbind(df_inst,
                 select(df_dwave, all_of(
                         c("logfile", "instance_id", "instance_type",
                           "sol_time", "obj_from_QPU_sol"))))

df_quera <- read.csv("./run_logs/summaries/quera_summary.csv")
df_quera <- filter(df_quera, success == "True" & (!is.na(obj_from_QPU_sol)))

df_inst <- rbind(df_inst,
                 select(df_quera, all_of(
                         c("logfile", "instance_id", "instance_type",
                           "sol_time", "obj_from_QPU_sol"))))

str(df_inst)

df_inst = plyr::rename(df_inst, c("sol_time" = "QPU_soltime"))

solved_inst <- df_inst %>% filter(instance_type == "MWC") %>%
  pull(instance_id) %>%
  unique()

length(solved_inst)

df_s <- filter(df, instance_id %in% solved_inst)

all_inst <- read.csv("./run_logs/instances.csv")

df_s = merge(x=df_s, y = select(all_inst, c("id", "qubo_vars")),
             by.x="instance_id", by.y="id")

nrow(df_s)

# A figure for cases when both approaches were optimal

df_simpler = filter(df_s, (status_QUBO == 2) & (status_LBOP) == 2)
df_harder = filter(df_s, (status_QUBO != 2) | (status_LBOP !=2))

ggplot(df_simpler) +
  geom_point(aes(x = sol_time_QUBO, y = sol_time_LBOP,
                 color=qubo_vars),
             size=5, alpha=0.8) +
  geom_abline(slope = 1.0, intercept = 0.0, color = "red", size=1) +
  xlab("Runtime for QUBO formulation, seconds") +
  ylab("Runtime for LBOP formulation, seconds") +
  theme(
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    axis.title.x = element_text(size = 30, margin = margin(t=20)),
    axis.title.y = element_text(size = 30, margin = margin(r = 20)),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    panel.background = element_rect(fill = NA, color = "black"),
    legend.key=element_blank(),
    legend.background=element_rect(fill='white'),
    legend.direction = "horizontal",
    legend.position = c(0.65,0.9),
    legend.text = element_text(size=30),
    legend.title = element_text(size=30, margin = margin(r=20)))+
  scale_colour_viridis_c(name="No. variables", # option="plasma",
                         breaks=with(df_simpler,
                                     seq(from=min(qubo_vars), to=max(qubo_vars), length.out=3)))

ggsave("./figures/LBOPvsQUBO_sec.png", width=10, height=10)

# A figure for cases when at least one formulation timed out
ggplot(df_harder) +
  geom_point(aes(x = gap_QUBO*100, y = gap_LBOP*100, color=qubo_vars),
             size=5, alpha=0.8) +
  geom_abline(slope = 1.0, intercept = 0.0, color = "red", size=2) +
  xlab("Optimality gap for QUBO formulation") +
  ylab("Optimality gap for LBOP formulation") +
  theme(
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    axis.title.x = element_text(size = 30, margin = margin(t=20)),
    axis.title.y = element_text(size = 30, margin = margin(r = 20)),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    panel.background = element_rect(fill = NA, color = "black"),
    legend.key=element_blank(),
    legend.background=element_rect(fill='white'),
    legend.direction = "horizontal",
    legend.position = c(0.65,0.9),
    legend.text = element_text(size=30),
    legend.title = element_text(size=25))+
  scale_colour_viridis_c(name="No. variables", #option="plasma",
                         breaks=with(df_simpler,
                                     seq(from=min(qubo_vars), to=max(qubo_vars), length.out=3)))

ggsave("./figures/LBOPvsQUBO_gap.png", width=10, height=10)
