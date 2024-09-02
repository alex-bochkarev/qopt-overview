suppressPackageStartupMessages({
  library(ggplot2)
  library(tidyr)
  library(dplyr)
  library(forcats)
  library(optparse)
  library(stringr)
})

option_list = list(
    make_option(c("-i", "--input"), type="character", default=NULL,
                help="input sample .csv", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="./out.png",
                help="output file name for the figure[default= %default]", metavar="character"),
    make_option(c("-O", "--objective"), action="store_true", default=FALSE,
                help="create objective figure only[default= %default]"),
    make_option(c("-s", "--solutions"), type="character", default="NA",
                help="solutions csv file [default= %default]", metavar="character"),
    make_option(c("-t", "--logtype"), type="character", default="dwave",
                help="CSV log type [default= %default]", metavar="character")
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
sols = opt$solutions

logtype = opt$logtype

if (sols != "NA") {
  dfs = read.csv(sols, stringsAsFactors = TRUE)
  if ("inst_id" %in% colnames(dfr)){
    dfr$inst_id = as.character(dfr$inst_id)
    opt_obj = dfs$objective[dfs$instance_id == dfr$inst_id[[1]]]
    if (is.numeric(opt_obj) & (length(opt_obj) == 1)) {
    draw_classic_solution = TRUE
    } else {
      print(paste("Instance id", dfr$inst_id[[1]], " but no objective in", sols))
      draw_classic_solution = FALSE
    }
  }
}

df <- select(dfr, setdiff(colnames(dfr), c("X", "inst_id", "logfile", "inst_type", "inst_size")))
df$solution = forcats::fct_reorder(as.factor(df$solution), df$objective)

if ("sol_feasible" %in% colnames(df)) {
  df$sol_feasible = factor(df$sol_feasible, levels = c("True", "False"),
                           labels=c("Yes", "No"))
}

df_named = rename(df,
                  all_of(c(Feasible       = "sol_feasible"    ,
                           `sol. count`   = "no_occ"          ,
                           `QUBO obj.`    = "objective"       ,
                           `orig. obj.`   = "orig_obj"        )))

if ("chain_break_frac" %in% colnames(df)) {
  df_named = rename(df_named, `ch. breaks` = "chain_break_frac")
}

pivot_cols = setdiff(colnames(df_named), c("solution", "Feasible"))

dfl <- pivot_longer(df_named,
                    cols=all_of(pivot_cols), names_to="var", values_to="val")

dfl$varf = factor(dfl$var, levels=c("sol. count", setdiff(pivot_cols, c("sol. count"))))

titlestr = dfr$inst_id[[1]]

if (logtype=="dwave"){
  titlestr = paste0(titlestr, ", ", dfr$inst_size[[1]], " QUBO variables.")
}else if (logtype=="quera"){
  titlestr = paste0(titlestr, ", ", str_length(dfr$solution[[1]]), " QUBO variables.")
}

p = ggplot(dfl, mapping=aes(x=solution, y=val))+
  geom_col(data = dplyr::filter(dfl, var == "sol. count"))+
  geom_col(aes(fill=Feasible))+
  scale_fill_manual(values = c("#21918c", "#fde725"))+
  facet_grid(varf ~ ., scales = "free_y")+
  theme(
    axis.text.x = element_blank(), #element_text(size = 5, angle=90),
    axis.text.y = element_text(size = 13),
    axis.title.x = element_text(size = 26),
    axis.title.y = element_blank(),
    panel.background = element_rect(fill = NA, color = "black"),
    panel.grid.major = element_blank(),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22, angle = 90),
    strip.background = element_blank(),
    title = element_text(size=22),
    legend.title = element_text(size=22),
    legend.text = element_text(size=22),
    legend.position="inside",
    legend.position.inside = c(0.8,0.95)
  )+
  ylab("Values")+
  ggtitle(titlestr)

if (draw_classic_solution) {
  p = p + geom_hline(data = dfl %>% filter(var %in% c("orig. obj.")),
                     aes(yintercept = opt_obj), color='#440154', linewidth = 2,
                     linetype='solid')
}
p

ggsave(paste0(opt$out, ".png"), width = 16, height = 10)
