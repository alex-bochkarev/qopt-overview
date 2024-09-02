suppressPackageStartupMessages({
  library(ggplot2)
  library(tidyr)
  library(dplyr)
  library(forcats)
  library(optparse)
})
option_list = list(
    make_option(c("-i", "--input"), type="character", default=NULL,
                help="input sample .csv", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="./out.png",
                help="output file name for the figure[default= %default]", metavar="character"),
    make_option(c("-O", "--objective"), action="store_true", default=FALSE,
                help="create objective figure only[default= %default]"),
    make_option(c("-s", "--solutions"), type="character", default="NA",
                help="solutions csv file [default= %default]", metavar="character")
);

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$input) | is.null(opt$out)){
    print_help(opt_parser)
    stop("Please specify both input and output files", call.=FALSE)
}

infile=opt$input


col.classes = c("num", "character")
names(col.classes) <- c(".default", "solution")

dfr <- read.csv(infile, stringsAsFactors = TRUE,
               colClasses=col.classes)

draw_classic_solution = FALSE

if (opt$solutions != "NA") {
  dfs = read.csv(opt$solutions, stringsAsFactors = TRUE)
  if ("inst_id" %in% colnames(dfr)){
    dfr$inst_id = as.character(dfr$inst_id)
    opt_obj = dfs$objective[dfs$instance_id == dfr$inst_id[[1]]]
    if (is.numeric(opt_obj) & (length(opt_obj) == 1)) {
    draw_classic_solution = TRUE
    } else {
      print(paste("Instance id", dfr$inst_id[[1]], " but no objective in", opt$solutions))
      draw_classic_solution = FALSE
    }
  }
}


draw_fig <- function(samples_df, title){
  df <- select(samples_df, setdiff(colnames(samples_df), c("X", "inst_id", "logfile", "inst_type", "inst_size")))
  df$solution = forcats::fct_reorder(as.factor(df$solution), df$objective)

  if ("sol_feasible" %in% colnames(df)) {
    df$sol_feasible = factor(df$sol_feasible, levels = c("True", "False"),
                             labels=c("Yes", "No"))
  }

  pivot_cols = setdiff(colnames(df), c("solution", "sol_feasible"))


  dfl <- pivot_longer(df, cols=all_of(pivot_cols), names_to="var", values_to="val")

  dfl$varf = factor(dfl$var, levels=c("no_occ", setdiff(pivot_cols, c("no_occ"))))

  p = ggplot(dfl, mapping=aes(x=solution, y=val))+
    geom_col(data = dplyr::filter(dfl, var == "no_occ"), fill='blue')+
    geom_col(aes(fill=sol_feasible))+
    scale_fill_manual(values = c("darkgreen", "red"))+
    facet_grid(varf ~ ., scales = "free_y")+
    theme(
      axis.text.x = element_blank(), #element_text(size = 5, angle=90),
      axis.text.y = element_text(size = 13),
      axis.title.x = element_text(size = 26),
      axis.title.y = element_text(size = 26, margin = margin(t = 50)),
      panel.background = element_rect(fill = NA, color = "black"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      ## element_line(
      ##   linewidth = 0.5, linetype = "solid",
      ##   color = "lightgrey"
      ## ),
      strip.text.x = element_text(size = 22),
      strip.text.y = element_text(size = 22, angle = 90),
      strip.background = element_blank()
    )+
    ylab("Values")+
    ggtitle(title)

  if (draw_classic_solution) {
    p = p + geom_hline(data = dfl %>% filter(var %in% c("orig_obj")),
                       aes(yintercept = opt_obj), color='blue', linewidth = 2,
                       linetype='dashed')
  }
  p
}

if ("inst_id" %in% colnames(dfr)){
  title = paste0(infile," → ", dfr$inst_id[[1]])
} else {
  title = infile
}

draw_fig(dfr, title)
ggsave(paste0(opt$out, ".png"), width = 16, height = 10)
