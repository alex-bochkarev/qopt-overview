suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(viridisLite)
  library(latex2exp)
})

df = read.csv("./run_logs/classic_solutions/MWC_bm_gurobi.csv")

df <- df %>%
  rename(t_gurobi = runtime,
         obj_gurobi = objective)

dwave_df = read.csv("./run_logs/summaries/dwave_summary.csv")
idf = read.csv("./run_logs/MWC_inst_summary.csv")

sel_inst = left_join(
  x = select(df, c("instance_id", "qubo_vars", "obj_gurobi", "t_gurobi")),
  y = select(dwave_df, c("instance_id", "sol_time", "obj_from_QPU_sol")),
  by = "instance_id")

sel_inst <- sel_inst %>%
  rename(t_QPU = sol_time,
         obj_QPU = obj_from_QPU_sol)

sel_inst <- merge(x=sel_inst, y = idf, by.x="instance_id",
                  by.y="id")

ggplot(sel_inst) +
  geom_point(aes(x = qubo_vars, y = (obj_QPU - obj_gurobi)/obj_gurobi,
                 color=as.factor(p), shape=as.factor(p)), size=5)+
  labs(
    x=TeX("No. of binary (QUBO) variables, $N$"),
    y=TeX("Relative objective deviation, $R_f"))+
    theme(
        plot.title=element_text(size=25),
        axis.text.x = element_text(size = 25),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 30),
        axis.title.y = element_text(size = 30, margin = margin(t = 50)),
        panel.background = element_rect(fill = NA, color = "black"),
        panel.grid.major = element_line(
        size = 0.5, linetype = "solid",
        color = "lightgrey"
        ),
      legend.key = element_blank(),
      legend.text = element_text(size=25),
      legend.title = element_text(size=25)
      )+
  scale_color_viridis_d()+
  labs(shape = "ER(p):", color = "ER(p):")

ggsave("./figures/gurobi_timeout.png", width=12, height = 10)

ggplot(sel_inst) +
  geom_histogram(aes(x = (obj_QPU - obj_gurobi)/obj_gurobi), size=3)+
  ## xlab("Problem size (QUBO variables)")+
  ## ylab("Relative objective deviation Rf")+
    theme(
        plot.title=element_text(size=25),
        axis.text.x = element_text(size = 25),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 25),
        axis.title.y = element_text(size = 25, margin = margin(t = 50)),
        panel.background = element_rect(fill = NA, color = "black"),
        panel.grid.major = element_line(
        size = 0.5, linetype = "solid",
        color = "lightgrey"
        ))
