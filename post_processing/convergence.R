suppressPackageStartupMessages({
    library(ggplot2)
    library(optparse)
    library(dplyr)
})

option_list = list(
    make_option(c("-i", "--input"), type="character", default=NULL,
                help="input sample .csv", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="./out.png",
                help="output file name for the figure[default= %default]", metavar="character"))

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$input) | is.null(opt$out)){
    print_help(opt_parser)
    stop("Please specify both input and output files", call.=FALSE)
}

print(paste("Drawing a figure for file: ", opt$input))
df = read.csv(opt$input)

logfile = unique(df$logfile)
inst_id = unique(df$instance_id)

# ggplot(df)+
ggplot(dplyr::filter(df, !(variable %in% c("shots", "timestamp"))))+
  geom_line(aes(x=iteration, y=value))+
  facet_wrap(. ~ variable, ncol = 1,
             scales = "free_y")+
  theme(
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 13),
    axis.title.x = element_text(size = 26),
    axis.title.y = element_blank(),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      linewidth = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = -90),
    strip.background = element_blank()
  )+
  ggtitle(paste0(logfile, ": ", inst_id))+
  xlab("Iteration number")

ggsave(opt$out, height=16, width=10)
