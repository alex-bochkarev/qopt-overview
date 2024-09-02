library(dplyr)


get_success_rates = function(fname) {
    df = read.csv(fname, stringsAsFactors = FALSE)
    success_per_inst = df %>% group_by(instance_id) %>%
    summarise(
        got_objective = sum(!is.na(obj_from_QPU_sol)),
        success_rate = got_objective / n())

    return(success_per_inst)
}

SR_crosscheck = function(fname) {
    df = read.csv(fname, stringsAsFactors = FALSE)
    return(sum(!is.na(df$obj_from_QPU_sol)) / nrow(df))
}

print("Mean success rates per instance are:")
df = get_success_rates("./run_logs/summaries/dwave_summary_full.csv")
print(paste0("DWave: ", as.character(mean(df$success_rate))))
df = get_success_rates("./run_logs/summaries/ibm-qpu_summary_full.csv")
print(paste0("IBM-QPU: ", as.character(mean(df$success_rate))))
df = get_success_rates("./run_logs/summaries/ibm-sim_summary_full.csv")
print(paste0("IBM-sim: ", as.character(mean(df$success_rate))))
df = get_success_rates("./run_logs/summaries/quera_summary_full.csv")
print(paste0("Quera: ", as.character(mean(df$success_rate))))

print("Mean success rate (all successes / all runs):")
print(paste0("DWave: ", SR_crosscheck("./run_logs/summaries/dwave_summary_full.csv")))
print(paste0("IBM-QPU: ", SR_crosscheck("./run_logs/summaries/ibm-qpu_summary_full.csv")))
print(paste0("IBM-sim: ", SR_crosscheck("./run_logs/summaries/ibm-sim_summary_full.csv")))
print(paste0("QuEra: ", SR_crosscheck("./run_logs/summaries/quera_summary_full.csv")))
