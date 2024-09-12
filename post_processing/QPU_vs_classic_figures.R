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

# Load classic solutions data

df_TSP <- read.csv("./run_logs/classic_solutions/TSP_MTZ.csv")
df_UDMIS <- read.csv("./run_logs/classic_solutions/UDMIS.csv")
df_MWC <- read.csv("./run_logs/classic_solutions/MWC_QUBO.csv")

df_classic <- rbind(select(df_TSP, -c("tour")),
                    select(df_UDMIS, -c("solution")))


df_MWC$objective_QUBO = - df_MWC$objective_QUBO  # because we were minimizing

df_classic <- rbind(df_classic,
                    dplyr::rename(select(df_MWC, -c("solution_QUBO")),
                                  c("instance_id" = "instance_id",
                                    "sol_time" = "sol_time_QUBO",
                                    "status" = "status_QUBO",
                                    "objective" = "objective_QUBO",
                                    "gap" = "gap_QUBO")))

# Load quantum solutions data
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
df_quera$device = rep("NA-OPT", length(df_quera$logfile))

df_inst <- rbind(df_inst,
                 select(df_quera, all_of(
                         c("device", "logfile", "instance_id", "instance_type",
                           "sol_time", "obj_from_QPU_sol"))))

str(df_inst)

df_inst = plyr::rename(df_inst, c("sol_time" = "QPU_soltime"))

dfm = merge(x = df_inst, y = dplyr::rename(df_classic,
                                           c("classic_sol_time"="sol_time",
                                              "classic_objective"="objective",
                                              "classic_status"="status")),
            by = "instance_id")


dfm$device = ifelse(dfm$device %in% c("ibm_cusco","ibm_nazca"), "QAOA-OPT", dfm$device)
dfm$device = ifelse(dfm$device=="Advantage_system4.1", "QA-OPT", dfm$device)
dfm$device = ifelse(dfm$device=="ibmq_qasm_simulator", "SIM-OPT", dfm$device)
dfm$device = ifelse(dfm$device=="QuEra", "NA-OPT", dfm$device)


dfm$prob_type = ifelse(dfm$instance_type=="MWC", "MaxCut",
                      ifelse(dfm$instance_type=="UDMIS", "UD-MIS", dfm$instance_type))

ylabs = c(0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100, 300, 1200)
xlabs = c(0.01, 0.1, 1, 5, 10, 25, 50, 100, 250, 1000, 5000, 25000)

dfm$ProbType <- factor(dfm$prob_type,
                          levels = c("UD-MIS", "MaxCut", "TSP"))

ggplot(dfm) +
  geom_point(aes(x = QPU_soltime, y = classic_sol_time,
                 color=ProbType,
                            shape=device), size=5, alpha=0.7) +
  geom_abline(slope = 1.0, intercept = 0.0, color = "red", size=2) +
  xlab("Total runtime (QPU), seconds") +
  ylab("Baseline runtime (Gurobi), seconds") +
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
    legend.position = c(0.875,0.7),
    legend.text = element_text(size=30),
    legend.title = element_text(size=30))+
  scale_colour_viridis_d(name="Problem type")+
  scale_shape_discrete(name="Device")+
  scale_x_continuous(breaks = xlabs,
                     labels = xlabs,
                     trans = scales::log_trans())+
  scale_y_continuous(breaks = ylabs,
                     labels = ylabs,
                     trans = scales::log_trans())

ggsave("./figures/qpu_vs_cpu_runtimes.png", width = 15, height = 10)

# Let's consider relative deviations
dfm$QPU_rel_obj_signed = with(dfm,
    (obj_from_QPU_sol - classic_objective) / classic_objective)

dfm$QPU_rel_runtime_signed = with(dfm,
    (QPU_soltime - classic_sol_time) / classic_sol_time)

dfm = filter(dfm, instance_id != "MWC1")  # that's a very degenerate case

df_hard = filter(dfm, classic_sol_time > 20)

# Note this is only larger MaxCuts, solved by D-Wave only:
unique(df_hard$prob_type)
unique(df_hard$device)

ggplot(df_hard)+
  geom_vline(xintercept = 0.0, color='red', size=1)+
  geom_hline(yintercept = 0.0, color='red', size=1)+
  geom_point(aes(x = QPU_rel_runtime_signed, y = QPU_rel_obj_signed), size=2,
              width=0.1)+
  xlab("Relative runtime deviation") +
  ylab("Relative objective deviation") +
  theme(
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 13),
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18, margin = margin(t = 50)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = -90),
    strip.background = element_blank()
  )

ggsave("./figures/hard_instances.png", width = 5, height = 5)

# Generate summary tables
options(dplyr.width = Inf)

dfm$no_worse_than_baseline = with(dfm,
                                  ifelse(instance_type %in% c("MWC", "UDMIS"),
                                         QPU_rel_obj_signed >= 0,
                                         QPU_rel_obj_signed <= 0))


dfm$QPUType <- factor(dfm$device,
                          levels = c("NA-OPT", "QA-OPT", "QAOA-OPT", "SIM-OPT"))

df_summary = dfm %>%
  dplyr::group_by(ProbType, QPUType) %>%
  dplyr::summarise(
    Total = n(),
    w25 = sum(abs(.data[["QPU_rel_obj_signed"]]) <= 0.25),
    w25Share = w25*100 / Total,
    w10= sum(abs(.data[["QPU_rel_obj_signed"]]) <= 0.1),
    w10Share = w10*100 / Total,
    w5= sum(abs(.data[["QPU_rel_obj_signed"]]) <= 0.05),
    w5Share = w5*100 / Total,
    NoWorse = sum(.data[["no_worse_than_baseline"]]),
    NoWorseShare = NoWorse*100 / Total)
df_summary

latex(df_summary, file="./figures/qpu_vs_cpu.tex", digits=3)

######################################################################
## ad-hoc analysis for QuEra dataset
## dfq = filter(dfm, device == "QuEra QPU")

## inst_stats = read.csv("./instances/UDMIS_inst_stats.csv")

## dfq = merge(dfq, inst_stats, by = "instance_id")
## dfq$abs_dev = with(dfq, classic_objective - obj_from_QPU_sol)

## ggplot(dfq)+
##   geom_point(aes(x=E, y=abs_dev, color=factor(R)))
