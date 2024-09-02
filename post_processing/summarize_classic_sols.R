library(dplyr)

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

write.csv(df_classic, "./run_logs/classic_solutions/solutions_all.csv")
