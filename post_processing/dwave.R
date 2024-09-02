suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(forcats)
  library(viridis)
  library(stargazer)
})

df <- read.csv("./run_logs/summaries/dwave_summary.csv", stringsAsFactors = FALSE)

df$qubit_cost = df$emb_qubits / df$binary_vars

reg_MWC = lm(log(emb_qubits) ~ log(binary_vars),
         data=filter(df, instance_type == "MWC"))
summary(reg_MWC)
reg_MWC$coefficients

reg_TSP = lm(log(emb_qubits) ~ log(binary_vars),
         data=filter(df, instance_type == "TSP"))
summary(reg_TSP)
reg_TSP$coefficients

reg_UDMIS = lm(log(emb_qubits) ~ log(binary_vars),
         data=filter(df, instance_type == "UDMIS"))
summary(reg_UDMIS)
reg_UDMIS$coefficients

# regression tables
stargazer(reg_UDMIS, reg_MWC, reg_TSP,
          dep.var.labels = c("$\\log(Q_e)$, in different regressions:"),
          covariate.labels = c("$\\beta$: $\\log(Q_{\\text{QUBO}})$",
                               "$\\beta_0$: $\\log \\text{Const}$"),
          column.labels = c("UD-MIS", "MaxCut", "TSP"),
          out="./figures/regression_summary.tex")

df$prob_type = ifelse(df$instance_type=="MWC", "MaxCut",
                      ifelse(df$instance_type=="UDMIS", "UD-MIS", df$instance_type))

df$prob_type = factor(df$prob_type, levels = c("UD-MIS", "MaxCut", "TSP"))

labs = c(5, 10, 25, 50, 75, 100, 150, 200, 500, 1000)

ggplot(df) +
  geom_jitter(aes(x = binary_vars, y = emb_qubits, color=prob_type,
                  shape=prob_type), width=0.02, size=5, alpha=0.5)+
  geom_line(aes(x = binary_vars,
                y = exp(coef(reg_MWC)[1] + coef(reg_MWC)[2]*log(binary_vars))), color='#440154FF',
            linetype='dashed')+
  geom_line(aes(x = binary_vars,
                y = exp(coef(reg_TSP)[1] + coef(reg_TSP)[2]*log(binary_vars))), color='#21908CFF',
            linetype='dashed')+
  geom_line(aes(x = binary_vars,
                y = exp(coef(reg_UDMIS)[1] + coef(reg_UDMIS)[2]*log(binary_vars))), color='#FDE725FF',
            linetype='dashed')+
  theme(
    axis.text.x = element_text(size = 25),
    axis.text.y = element_text(size = 25),
    axis.title.x = element_text(size = 25),
    axis.title.y = element_text(size = 25, margin = margin(t = 50)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = 180),
    strip.background = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(0.2, 0.8),
    legend.background=element_rect(fill='white'),
    legend.key=element_blank(),
    legend.text = element_text(size=25),
    legend.title = element_text(size=25))+
  scale_color_viridis_d(name="Problem type")+
  scale_shape_discrete(name="Problem type")+
  scale_x_continuous(breaks = labs,
                     labels = labs,
                     trans = scales::log_trans())+
  scale_y_continuous(breaks = labs,
                     labels = labs,
                     trans = scales::log_trans())+
  xlab("Number of logical qubits")+
  ylab("Number of physical qubits")

ggsave("./figures/dwave_qubit_cost_logs_regline.png", width = 15, height = 10)

edf = read.csv("./run_logs/summaries/embeddings.csv", stringsAsFactors = TRUE)

df_we = merge(x=df, y = select(edf, c("instance_id", emb_time, emb_success)), by = "instance_id")

df_we$true_embedding_time = with(df_we,
                                 ifelse(embedding_time > 0, embedding_time, emb_time))

df_we$true_total_time = with(df_we,
                             ifelse(embedding_time > 0, sol_time, sol_time + emb_time))

ggplot(df_we)+
  geom_jitter(aes(x = binary_vars, y= 100*(true_embedding_time / true_total_time),
                  color=prob_type, shape=prob_type),
              width = 1, size=5
  ) +
  xlab("Instance size, binary vars (before embedding)") +
  ylab("Share of embedding time, %") +
  scale_y_continuous(
    labels = scales::number_format(accuracy = 1),
    breaks = seq(0, 90, 10))+
  theme(
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30, margin = margin(r = 20)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    legend.key=element_blank(),
    legend.background=element_rect(fill='white'),
    legend.position = c(0.2, 0.8),
    legend.text = element_text(size=30),
    legend.title = element_text(size=30))+
  scale_color_viridis_d(name="Problem type")+
  scale_shape_discrete(name="Problem type")
  # scale_x_continuous(breaks = seq(4, 20, 2), limits = c(4, 20))

ggsave("./figures/dwave_embtime_share.png", width = 15, height = 10)

# stacked plot for runtimes
df_we$noemb_time = with(df_we,
                        true_total_time - true_embedding_time)

dfl = pivot_longer(filter(df_we, instance_type=="TSP"), cols = c("emb_time", "noemb_time"), names_to = "time_cat", values_to = "time_val")

dfl$f_file = forcats::fct_reorder(as.factor(dfl$filename), dfl$binary_vars)

ggplot(dfl, aes(fill=time_cat, x = f_file, y = time_val))+
  geom_bar(position="stack", stat="identity")

df = filter(df, success == "True")

sample_1 = function(dfrm){
  do.call( rbind, lapply( split(dfrm, dfrm$instance_id) ,
                         function(dfx) dfx[sample(nrow(dfx), 1) , ] )
          )
}

ggplot(sample_1(df))+
  geom_jitter(aes(x=binary_vars, y=embedding_time / sol_time,
                  size=sol_time, color=sol_time),
              width=1)

sample_Nsize = function(dfrm, N){
  do.call( rbind, lapply( split(dfrm, dfrm$binary_vars) ,
                         function(dfx) dfx[sample(nrow(dfx), N) , ] )
          )
}

draw_barplot <- function(inst_type, Nsamples){
  dfs = sample_Nsize(filter(df_we, instance_type==inst_type), Nsamples)

  dfl = pivot_longer(dfs, cols = c("noemb_time", "true_embedding_time"), names_to = "time_cat", values_to = "time_val")
  dfl$caption = paste0(dfl$instance_id, " (",dfl$binary_vars," vars)")
  dfl$f_file = forcats::fct_reorder(as.factor(dfl$filename), dfl$binary_vars)

  dfl$time_cat = recode(dfl$time_cat,
                        noemb_time = "annealing time",
                        true_embedding_time = "embedding time")

  ggplot(dfl, aes(fill=time_cat, x = f_file, y = time_val))+
    geom_bar(position="stack", stat="identity")+
    facet_grid(. ~ binary_vars, space="free", scales="free", switch="x")+
    theme(
      axis.text.x = element_blank(), #element_text(size = 7, angle=90),
      axis.text.y = element_text(size = 35),
      axis.title.x = element_blank(), #element_text(size = 25),
      axis.title.y = element_blank(), #element_text(size = 25, margin = margin(t = 50)),
      panel.background = element_rect(fill = NA, color = "black"),
      panel.grid.major = element_line(
        size = 0.5, linetype = "solid",
        color = "lightgrey"
      ),
      strip.text.x = element_text(size = 35),
      strip.text.y = element_text(size = 22, angle = 180),
      strip.background = element_blank(),
      legend.position = c(0.2,0.9),
      legend.text = element_text(size=30),
      legend.title = element_text(size=30))+
    ## xlab("Number of binary variables")+
    ## ylab("Runtimes: embedding + annealing, seconds")+
    scale_fill_viridis_d("Runtime component",
                         direction=-1)
}

draw_barplot("TSP", 3)
ggsave("./figures/runtime_vs_embtime_TSP.png", height = 10, width = 15)

draw_barplot("MWC", 10)
ggsave("./figures/runtime_vs_embtime_MWC.png", height = 10, width = 15)

draw_barplot("UDMIS", 1)
ggsave("./figures/runtime_vs_embtime_UDMIS.png", height = 10, width = 15)

ggplot(edf)+
  geom_jitter(aes(x=log(qubo_vars), y=log(emb_time),
                 color=instance_type, shape=instance_type),
             width=0.01, size=3, alpha=0.7)+
    theme(
      axis.text.x = element_text(size = 15),
      axis.text.y = element_text(size = 15),
      axis.title.x = element_text(size = 26),
      axis.title.y = element_text(size = 26, margin = margin(t = 50)),
      panel.background = element_rect(fill = NA, color = "black"),
      panel.grid.major = element_line(
        size = 0.5, linetype = "solid",
        color = "lightgrey"
      ),
      legend.position = c(0.2,0.9),
      legend.text = element_text(size=20),
      legend.title = element_text(size=20))+
  scale_color_viridis_d()

ggsave("./figures/embeddings_scaling_long.png", width = 16, height = 10)

ggplot(df_we)+
  geom_jitter(aes(x = binary_vars, y= true_total_time - true_embedding_time,
                  color=prob_type, shape=prob_type),
              width = 1, size=5
  ) +
  xlab("Instance size, binary variables (before embedding)") +
  ylab("Annealing time, seconds") +
  scale_y_continuous(
    labels = scales::number_format(accuracy = 1),
    breaks = seq(0, max(df_we$true_total_time - df_we$true_embedding_time, na.rm=TRUE),
                 2))+
  theme(
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30, margin = margin(r = 20)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      size = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = 180),
    strip.background = element_blank(),
    legend.key=element_blank(),
    legend.background=element_rect(fill='white'),
    legend.position = c(0.2, 0.8),
    legend.text = element_text(size=30),
    legend.title = element_text(size=30))+
  scale_color_viridis_d(name="Problem type")+
  scale_shape_discrete(name="Problem type")

ggsave("./figures/dwave_soltimes_noemb.png", width = 15, height = 10)
