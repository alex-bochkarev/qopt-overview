suppressPackageStartupMessages({
library(dplyr)
library(ggplot2)
})

qap = read.csv("./run_logs/classic_solutions/TSP.csv")
mtz = read.csv("./run_logs/classic_solutions/TSP_MTZ.csv")

inst = merge(x = qap, y = mtz, by="instance_id", suffixes = c(".qap", ".mtz"))

summary(qap$gap)
summary(mtz$gap)
summary(with(inst, objective.mtz - objective.qap))

ggplot(inst)+
  geom_abline(slope = 1.0, intercept = 0.0, color='red', size=2)+
  geom_point(aes(x = sol_time.qap, y = sol_time.mtz ))
