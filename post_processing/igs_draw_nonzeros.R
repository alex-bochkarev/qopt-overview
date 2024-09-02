suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(optparse)
  library(viridis)
})

df_full <- read.csv("./run_logs/instances.csv", stringsAsFactors = FALSE)
df_stats <- read.csv("./run_logs/summaries/all_devices_stats_full.csv", stringsAsFactors = FALSE)

df = merge(x = df_full, y = select(df_stats, instance_id), by.x="id", by.y="instance_id")

## ggplot(df_full)+
##   geom_histogram(aes(x = Qn_nonzeros), color = "lightgray", fill = "blue", bins = 30)+
##   facet_wrap(. ~ type) +
##   theme(
##     axis.text.x = element_text(size = 18, angle=90),
##     axis.text.y = element_text(size = 13),
##     axis.title.x = element_text(size = 26),
##     axis.title.y = element_text(size = 26, margin = margin(t = 50)),
##     panel.background = element_rect(fill = NA, color = "black"),
##     panel.grid.major = element_line(
##       linewidth = 0.5, linetype = "solid",
##       color = "lightgrey"
##     ),
##     strip.text.x = element_text(size = 22),
##     strip.text.y = element_text(size = 22, angle = 180),
##     strip.background = element_blank()
##   )+
##   xlab("Number of nonzeroes in the QUBO matrix, '000")

## ggsave("./figures/nonzeros_hists_full.png", width = 16, height = 10)

df_full$prob_type = ifelse(df_full$type=="MWC", "MaxCut",
                      ifelse(df_full$type=="UDMIS", "UD-MIS", df_full$type))

df_full$prob_type = factor(df_full$prob_type,
                           levels = c("UD-MIS", "MaxCut", "TSP"))

ggplot(filter(df_full, qubo_vars < 150))+
  geom_jitter(aes(x = qubo_vars, y = Qn_nonzeros / (qubo_vars*qubo_vars),
                  color=prob_type, shape=prob_type),
              size=5, width=0.5)+
  scale_colour_viridis_d(name="Problem type")+
  scale_shape_discrete(name="Problem type")+
  theme(
    axis.text.x = element_text(size = 23),
    axis.text.y = element_text(size = 23),
    axis.title.x = element_text(size = 26),
    axis.title.y = element_text(size = 26, margin = margin(t = 50)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      linewidth = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = 180),
    strip.background = element_blank(),
    legend.text = element_text(size=25),
    legend.title = element_text(size=25),
    legend.position = "inside",
    legend.position.inside = c(0.8,0.875),
    legend.background=element_rect(fill='white'),
    legend.key=element_blank()
  )+
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))+
  xlab("Number of QUBO variables.")+
  ylab("Share of nonzeros in Q.")

ggsave("./figures/nonzeros_vs_size.png", width=15, height = 10)
