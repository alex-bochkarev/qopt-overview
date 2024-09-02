suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(optparse)
})

option_list <- list(
    make_option(c("-i", "--input"), type = "character", default=NULL,
                help = "input sample .csv", metavar = "character"),
    make_option(c("-o", "--out"), type = "character", default = "./out.png",
                help = "output file name for the figure[default= %default]", metavar = "character"))

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$input) || is.null(opt$out)) {
    print_help(opt_parser)
    stop("Please specify both input and output files", call.=FALSE)
}
infile <- opt$input
df <- read.csv(infile, stringsAsFactors = FALSE)

ggplot(df) +
  geom_histogram(aes(x = qubo_vars), color = 'lightgray', fill = 'blue', bins = 30)+
  facet_wrap(. ~ type ) +
  theme(
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 13),
    axis.title.x = element_text(size = 26),
    axis.title.y = element_text(size = 26, margin = margin(t = 50)),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_line(
      linewidth = 0.5, linetype = "solid",
      color = "lightgrey"
    ),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = 180),
    strip.background = element_blank()
  )

ggsave(opt$out, width=16, height=10)
