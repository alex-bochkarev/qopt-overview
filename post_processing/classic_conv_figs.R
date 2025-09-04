suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(viridisLite)
})

my_colors <- viridis(2, option="D")

plot_convpic <- function(instID, save=FALSE){
    cdf = read.csv(paste0("./run_logs/classic_solutions/gurobi_MWC_logs/", instID,".log"))

    p = ggplot(cdf) +
        geom_line(aes(x=work_time, y=-obj_incumbent), color=my_colors[1], size=2)+
        geom_line(aes(x=work_time, y=-obj_bestbd), color=my_colors[2], size=2)+
        xlab("Wall-clock time, seconds") +
        ylab("Objective values (bounds)")+
        geom_vline(data=filter(cdf, newsol=="H"),
                    aes(xintercept=work_time), color='red', size=2, linetype='dashed')+
        theme(
            plot.title=element_text(size=35),
            axis.text.x = element_text(size = 25),
            axis.text.y = element_text(size = 20),
            axis.title.x = element_text(size = 35),
            axis.title.y = element_text(size = 35, margin = margin(t = 50)),
            panel.background = element_rect(fill = NA, color = "black"),
            panel.grid.major = element_line(
            size = 0.5, linetype = "solid",
            color = "lightgrey"
            ))+
        ggtitle(paste("Instance:", instID,"| final gap:",tail(cdf$obj_gap, n=1)))

    if(save){
        ggsave(plot = p,
               filename = paste0("./figures/gurobi_conv/", instID,".png"), width = 10, height = 10)
    }
    return(p)
}

# Instance: MWC96, 55 vars
plot_convpic("MWC96", save=TRUE)

# Instance: MWC260, 85 vars
plot_convpic("MWC260", save=TRUE)

# Instance: MWC101, 75 vars
plot_convpic("MWC101", save=TRUE)

# The following is, perhaps, not too interesting
# for the supplements text

# Instance: MWC186, 95 vars
plot_convpic("MWC186")

# Instance: MWC174, 55 vars
plot_convpic("MWC174")
