##
# QOpt-overview: computational recipes for the numerical part of the paper.
#
# @file Makefile
# @version 1.0

.PHONY: figures tables code-docs new-instances clean-instances clean-UDMIS clean-TSP clean-MWC

# don't remove intermediate files
.SECONDARY:

# key directories

INSTS=./instances
ORIGS=$(INSTS)/orig
QUBOS=$(INSTS)/QUBO
RLOGS=./run_logs
SUMM=$(RLOGS)/summaries
FIGS=./figures
PP=./post_processing
CPUSOLS=$(RLOGS)/classic_solutions
SUPPLDIR=./public_gh_repo

######################################################################
# Figures
#

# All the "computable" figures, more or less in the order of appearance
# in the paper (modulo LaTeX floats placement)
figures: $(FIGS)/inst_summary.png \
	$(FIGS)/runtime_vs_embtime_TSP.png $(FIGS)/runtime_vs_embtime_MWC.png \
	$(FIGS)/runtime_vs_embtime_UDMIS.png \
	$(FIGS)/qpu_vs_cpu_runtimes.png $(FIGS)/hard_instances.png \
	$(FIGS)/nonzeros_vs_size.png $(FIGS)/dwave_qubit_cost_logs_regline.png \
	$(FIGS)/dwave-MWC159.hist.png $(FIGS)/dwave-TSP53.hist.png $(FIGS)/dwave-TSP82.hist.png \
	$(FIGS)/quera-UDMIS1TG.hist.png $(FIGS)/quera-UDMIS4TG.hist.png $(FIGS)/quera-UDMIS7TG.hist.png \
	$(FIGS)/ibm-sim-UDMIS1TG.hist.png $(FIGS)/ibm-sim-UDMIS4TG.hist.png $(FIGS)/ibm-sim-UDMIS7TG.hist.png \
	$(FIGS)/ibm-qpu-UDMIS1TG.hist.png $(FIGS)/ibm-qpu-UDMIS4TG.hist.png $(FIGS)/ibm-qpu-UDMIS7TG.hist.png \
	$(FIGS)/dwave_soltimes_noemb.png $(FIGS)/dwave_embtime_share.png $(FIGS)/runs_summary.png

tables: $(FIGS)/qpu_vs_cpu.tex $(FIGS)/regression_summary.tex

# Recipies for the figures
$(FIGS)/inst_summary.png: $(PP)/inst_summary.R \
	$(SUMM)/quera_summary.csv $(SUMM)/dwave_summary.csv $(SUMM)/embeddings.csv \
	$(SUMM)/ibm-qpu_summary.csv $(SUMM)/ibm-sim_summary.csv \
	$(CPUSOLS)/solutions_all.csv
	Rscript $<

$(FIGS)/runs_summary.png: $(PP)/inst_summary_by_inst.R \
	$(SUMM)/quera_summary_full.csv $(SUMM)/dwave_summary_full.csv $(SUMM)/embeddings.csv \
	$(SUMM)/ibm-qpu_summary_full.csv $(SUMM)/ibm-sim_summary_full.csv \
	$(CPUSOLS)/solutions_all.csv
	Rscript $<

$(FIGS)/runtime_vs_embtime_TSP.png $(FIGS)/runtime_vs_embtime_MWC.png \
	$(FIGS)/runtime_vs_embtime_UDMIS.png \
	$(FIGS)/dwave_soltimes_noemb.png $(FIGS)/dwave_embtime_share.png \
	$(FIGS)/dwave_qubit_cost_logs_regline.png &: \
	$(PP)/dwave.R $(SUMM)/dwave_summary.csv $(SUMM)/embeddings.csv
	Rscript $<

$(FIGS)/qpu_vs_cpu_runtimes.png $(FIGS)/hard_instances.png \
	$(FIGS)/regression_summary.tex $(FIGS)/qpu_vs_cpu.tex &: \
	$(PP)/QPU_vs_classic_figures.R \
	$(CPUSOLS)/TSP_MTZ.csv $(CPUSOLS)/UDMIS.csv $(CPUSOLS)/MWC_QUBO.csv \
	$(SUMM)/ibm-qpu_summary.csv $(SUMM)/ibm-sim_summary.csv \
	$(SUMM)/dwave_summary.csv $(SUMM)/embeddings.csv \
	$(SUMM)/quera_summary.csv
	Rscript $<

$(FIGS)/nonzeros_vs_size.png: $(PP)/igs_draw_nonzeros.R \
	$(RLOGS)/instances.csv $(SUMM)/all_devices_stats_full.csv
	Rscript $<

$(RLOGS)/instances.csv: $(PP)/dataset_summary.py
	python -m post_processing.dataset_summary summarize_src_QUBOs > $@

$(FIGS)/dwave-MWC159.hist.png $(FIGS)/dwave-TSP53.hist.png $(FIGS)/dwave-TSP82.hist.png \
	$(FIGS)/quera-UDMIS1TG.hist.png $(FIGS)/quera-UDMIS4TG.hist.png $(FIGS)/quera-UDMIS7TG.hist.png \
	$(FIGS)/ibm-sim-UDMIS1TG.hist.png $(FIGS)/ibm-sim-UDMIS4TG.hist.png $(FIGS)/ibm-sim-UDMIS7TG.hist.png \
	$(FIGS)/ibm-qpu-UDMIS1TG.hist.png $(FIGS)/ibm-qpu-UDMIS4TG.hist.png $(FIGS)/ibm-qpu-UDMIS7TG.hist.png &:\
	make_suppl_sample_figures.sh $(CPUSOLS)/solutions_all.csv \
	dwave_suppl_samples.ids quera_suppl_samples.ids \
	ibm-sim_suppl_samples.ids ibm-qpu_suppl_samples.ids \
	$(SUMM)/dwave_summary.csv $(SUMM)/quera_summary.csv \
	$(SUMM)/ibm-sim_summary.csv $(SUMM)/ibm-qpu_summary.csv
	./make_suppl_sample_figures.sh

# Recipies for the source data
#

$(SUMM)/all_devices_stats_full.csv: $(RLOGS)/summaries/dwave_stats_full.csv \
	$(RLOGS)/summaries/ibm-qpu_stats_full.csv $(RLOGS)/summaries/ibm-sim_stats_full.csv \
	$(RLOGS)/summaries/quera_stats_full.csv
	head -n 1 $(RLOGS)/summaries/dwave_stats_full.csv > $@ && \
	tail -n+2 -q $(RLOGS)/summaries/*_stats_full.csv  >> $@

$(CPUSOLS)/solutions_all.csv: $(PP)/summarize_classic_sols.R \
	$(CPUSOLS)/TSP_MTZ.csv $(CPUSOLS)/UDMIS.csv $(CPUSOLS)/MWC_QUBO.csv
	Rscript $<

$(SUMM)/dwave_summary.csv: $(SUMM)/dwave_summary_full.csv $(PP)/select_runs.py
	python -m post_processing.select_runs -i $< -o $@

$(SUMM)/quera_summary.csv: $(SUMM)/quera_summary_full.csv $(PP)/select_runs.py
	python -m post_processing.select_runs -i $< -o $@

$(SUMM)/ibm-qpu_summary.csv: $(SUMM)/ibm-qpu_summary_full.csv $(PP)/select_runs.py
	python -m post_processing.select_runs -i $< -o $@

# Note that we remove all the runs for IBM simulator for instances with more than
# 32 variables (they are assumed to fail, it was a limitation of the simulator)
$(SUMM)/ibm-sim_summary.csv: $(SUMM)/ibm-sim_summary_full.csv $(PP)/select_runs.py
	python -m post_processing.select_runs -M 32 -i $< -o $@

$(SUMM)/dwave_summary_full.csv $(SUMM)/dwave_stats_full.csv &: $(PP)/logparser.py
	python -m post_processing.logparser summarize_dwave $(RLOGS)/dwave \
		$(SUMM)/dwave_summary_full.csv $(SUMM)/dwave_stats_full.csv

$(SUMM)/ibm-sim_summary_full.csv $(SUMM)/ibm-sim_stats_full.csv &: $(PP)/logparser.py
	python -m post_processing.logparser summarize_ibm $(RLOGS)/ibm-sim \
		$(SUMM)/ibm-sim_summary_full.csv $(SUMM)/ibm-sim_stats_full.csv "IBM-sim"

$(SUMM)/ibm-qpu_summary_full.csv $(SUMM)/ibm-qpu_stats_full.csv &: $(PP)/logparser.py
	python -m post_processing.logparser summarize_ibm $(RLOGS)/ibm-qpu \
		$(SUMM)/ibm-qpu_summary_full.csv $(SUMM)/ibm-qpu_stats_full.csv "IBM-QPU"

$(SUMM)/quera_summary_full.csv $(SUMM)/quera_stats_full.csv &: $(PP)/logparser.py
	python -m post_processing.logparser summarize_quera $(RLOGS)/quera \
		$(SUMM)/quera_summary_full.csv $(SUMM)/quera_stats_full.csv

$(SUMM)/embeddings.csv: precompute_embeddings.py
	touch $@  # a dummy instruction (see below)

#	WARNING: a time-intensive operation!
#	a good candidate to be run on different machines / in batches, etc.
#	E.g.
# 		python -m precompute_embeddings -t 16 -T 5000 -N 50 \
# 		-l .$(RLOGS)/dwave/embeddings/complex_MWC.ids \
# 		-s ./run_logs/v2_dataset/embeddings/complex_MWC_stats.csv
# 	see precompute_embeddings.py for details

######################################################################
# Finding solutions classically
# WARNING: a time-intensive operation!

$(CPUSOLS)/TSP_MTZ.csv: classic_solve_TSPs_MTZ.py
	touch $@
#	python -m classic_solve_TSPs_MTZ | tee $(CPUSOLS)/TSP_MTZ.log

$(CPUSOLS)/UDMIS.csv: classic_solve_UDMISes.py
	touch $@
#	python -m classic_solve_UDMISes | tee $(CPUSOLS)/UDMIS.log

$(CPUSOLS)/MWC_QUBO.csv: classic_solve_MWC_QUBO_only.py
	touch $@
#	python -m classic_solve_MWC_QUBO_only | tee $(CPUSOLS)/MWC_QUBO.log

######################################################################
# Instance generation code (for reference)

new-instances: $(INSTS)/instances.list

$(INSTS)/instances.list: $(INSTS)/TSP.inst $(INSTS)/MWC.inst $(INSTS)/UDMIS.inst
	cat $^ > $@
######################################################################
# Generating instance files
# WARNING: potentially destructive operation!

$(INSTS)/MWC.inst: MWC_inst.py
	touch $@
#	python -m MWC_inst $(INSTS) && \
#	ls $(ORIGS)/MWC*.orig.json > $(ORIGS)/MWC.list && \
#	ls $(QUBOS)/MWC*.qubo.json > $(QUBOS)/MWC.list && \
#	echo "`ls $(ORIGS)/MWC*.json | wc -l` MWC instances, generated: `date +'%Y-%m-%d, %H:%M'`" > $(INSTS)/MWC.instances

$(INSTS)/TSP.inst: TSP_inst.py
	touch $@
# python -m TSP_inst $(INSTS) && \
#	ls $(ORIGS)/TSP*.orig.json > $(ORIGS)/TSP.list && \
#	ls $(QUBOS)/TSP*.qubo.json > $(QUBOS)/TSP.list && \
#	echo "`ls $(ORIGS)/TSP*.json | wc -l` TSP instances, generated: `date +'%Y-%m-%d, %H:%M'`" > $(INSTS)/TSP.instances

$(INSTS)/UDMIS.inst: MIS_inst.py
	touch $@
#	python -m MIS_inst --dataset large --instdir $(INSTS) && \
#		python -m MIS_inst --dataset small --instdir $(INSTS) && \
#		ls $(ORIGS)/UDMIS*.orig.json > $(ORIGS)/UDMIS.list && \
#		ls $(QUBOS)/UDMIS*.qubo.json > $(QUBOS)/UDMIS.list && \
#		echo "`ls $(ORIGS)/UDMIS*.json | wc -l` UDMIS instances, generated: `date +'%Y-%m-%d, %H:%M'`" > $@

######################################################################
# Technical: supplementary archive preparation
code-docs: $(INSTS)/orig_instances.zip $(INSTS)/QUBO_formulations.zip \
	$(RLOGS)/dwave/raw_logs.zip $(RLOGS)/quera/raw_logs.zip \
	$(RLOGS)/ibm-qpu/raw_logs.zip $(RLOGS)/ibm-sim/raw_logs.zip
	cd .. && \
	rm -rf $(SUPPLDIR) && mkdir -p $(SUPPLDIR)/docs && \
	cd docs && rm -rf ./source/autosummary && make html && cd .. && \
	rsync -arv ./docs/build/html/* $(SUPPLDIR)/docs/ && \
	cd comps && \
	rsync -av --prune-empty-dirs --include-from=../files-for-supplement.list ./ ../$(SUPPLDIR)/ && \
	cd ../$(SUPPLDIR)/run_logs/dwave && unzip ./embeddings.zip && rm ./embeddings.zip && unzip ./raw_logs.zip && rm ./raw_logs.zip && \
	cd ../ibm-qpu && unzip ./raw_logs.zip && rm ./raw_logs.zip && \
	cd ../ibm-sim && unzip ./raw_logs.zip && rm ./raw_logs.zip && \
	cd ../quera && unzip ./raw_logs.zip && rm ./raw_logs.zip && \
	cd ../.. && \
	cp ../docs/source/GH_README.md ./README.md && \
	touch ../../.nojekyll


######################################################################
# Cleaning
# WARNING: destructive operation!
clean:
	rm -f $(SUMM)/*stats*.csv
	rm -f $(SUMM)/*summary*.csv
	rm -f $(FIGS)/*.png
	rm -f $(FIGS)/*.tex
	rm -f $(RLOGS)/dwave/samples-csv/*
	rm -f $(RLOGS)/quera/samples-csv/*
	rm -f $(RLOGS)/ibm-qpu/samples-csv/*
	rm -f $(RLOGS)/ibm-sim/samples-csv/*

######################################################################
# Testing

tests:
				python -m pytest tests/

# end
