suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(optparse)
  library(viridis)
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

ggplot(df)+
  geom_histogram(aes(x = Qn_nonzeros / 1000), color = "lightgray", fill = "blue", bins = 30)+
  facet_wrap(. ~ type) +
  theme(
    axis.text.x = element_text(size = 18, angle=90),
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
  )+
  xlab("Number of nonzeroes in the QUBO matrix, '000")

ggsave(opt$out, width = 16, height = 10)

ggplot(filter(df, qubo_vars <=100))+
  geom_jitter(aes(x = qubo_vars, y = Qn_nonzeros / (qubo_vars*qubo_vars), color=type, shape=type),
              size=3, width=0.5)+
  scale_colour_viridis_d()+
  theme(
    axis.text.x = element_text(size = 18, angle=90),
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
    strip.background = element_blank(),
    legend.text = element_text(size=20),
    legend.title = element_text(size=20),
    legend.position = c(0.8,0.9),
  )+
  xlab("No. of QUBO variables.")+
  ylab("Share of nonzeros in Q.")

ggsave("./figures/v2_dataset/nonzeros.png", width=16, height = 10)
