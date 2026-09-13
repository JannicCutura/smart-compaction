.PHONY: install_tex install_py tex arxiv arxiv_pkg arxiv_pkg_check slides clean params generate mount_data snapshot restore extract compact label train evaluate violins violins_wide_from_pdf pipeline ablation ablation_plots compact_512 query_bench

VENV := .venv
PY := $(VENV)/bin/python
export JAVA_HOME := /usr/lib/jvm/java-21-amazon-corretto
DATA_DIR := /mnt/data

install_tex:
	sudo apt-get update
	sudo apt-get install -y texlive texlive-latex-extra texlive-fonts-recommended texlive-science texlive-publishers cm-super

install_py: $(VENV)/bin/activate

$(VENV)/bin/activate: pyproject.toml
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip setuptools
	$(VENV)/bin/pip install -e ".[dev]"
	touch $(VENV)/bin/activate

params: install_py
	$(PY) code/params.py --csv code/grid.csv

generate: params
	$(PY) code/generate.py --csv code/grid.csv

snapshot:
	@if [ -d "$(DATA_DIR)/warehouse_snapshot" ]; then echo "Snapshot already exists. Run 'make restore' first or remove it manually."; exit 1; fi
	@echo "Snapshotting warehouse..."
	cp -a $(DATA_DIR)/warehouse $(DATA_DIR)/warehouse_snapshot
	@echo "Snapshot saved to $(DATA_DIR)/warehouse_snapshot"

restore:
	@if [ ! -d "$(DATA_DIR)/warehouse_snapshot" ]; then echo "No snapshot found at $(DATA_DIR)/warehouse_snapshot"; exit 1; fi
	@echo "Restoring warehouse from snapshot..."
	rm -rf $(DATA_DIR)/warehouse
	mv $(DATA_DIR)/warehouse_snapshot $(DATA_DIR)/warehouse
	@echo "Restore complete."

extract: install_py
	$(PY) code/extract.py --csv code/grid.csv --out data/features.csv

compact: install_py
	$(PY) code/compact_runner.py

label: install_py
	$(PY) code/label.py

train: install_py
	$(PY) code/train.py

evaluate: install_py
	$(PY) code/evaluate.py

# Only the two feature ridge plots (paper portrait + 16:9 slide version);
# needs data/dataset.csv but not the trained models.
violins: install_py
	$(PY) code/evaluate.py --only-violins

# Same wide plot without the dataset: lifts the 17 KDE curves back out of the
# published vector figure paper/plots/feature_violins.pdf (see the script).
violins_wide_from_pdf: install_py
	$(PY) code/violins_from_pdf.py

pipeline: extract snapshot compact label train evaluate
	@echo "=== Pipeline complete ==="
	@echo "Features:  data/features.csv"
	@echo "Compaction: data/compaction.csv"
	@echo "Dataset:   data/dataset.csv"

ablation: install_py
	$(PY) code/ablation.py

ablation_plots: install_py
	$(PY) code/ablation_plots.py

compact_512: install_py
	$(PY) code/compact_runner.py --target-mb 512 --out data/compaction_512mb.csv --rollback-first

label_512: install_py
	$(PY) code/label.py --compaction data/compaction_512mb.csv --out data/dataset_512mb.csv

query_bench: install_py
	$(PY) code/tpch_query_bench.py 2>&1 | tee logs/tpch_query_bench.log

tex:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error paper.tex && bibtex paper && pdflatex -interaction=nonstopmode -halt-on-error paper.tex && pdflatex -interaction=nonstopmode -halt-on-error paper.tex

# arXiv preprint: same source plus the IEEE copyright notice required for
# posting an accepted paper. Outputs paper/paper-arxiv.pdf and leaves the
# camera-ready paper.pdf untouched.
ARXIV_SRC := "\def\ARXIV{}\input{paper}"
arxiv:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -jobname=paper-arxiv $(ARXIV_SRC) && bibtex paper-arxiv && pdflatex -interaction=nonstopmode -halt-on-error -jobname=paper-arxiv $(ARXIV_SRC) && pdflatex -interaction=nonstopmode -halt-on-error -jobname=paper-arxiv $(ARXIV_SRC)

# arXiv submission package. arXiv compiles the .tex itself, so it cannot pass
# the command-line \def that `arxiv` above uses -- the switch has to be baked
# into the source, and the .bbl has to ship (arXiv does not run BibTeX).
# See paper/make_arxiv_pkg.py. Depends on `arxiv` for an up-to-date .bbl.
arxiv_pkg: arxiv
	python3 paper/make_arxiv_pkg.py
	@echo "=== Upload paper/paper-arxiv-submission.tar.gz to arXiv ==="

# Verify the package compiles the way arXiv does: pdflatex x3, no bibtex,
# in a scratch copy so no pre-existing .aux file can mask a missing input.
arxiv_pkg_check: arxiv_pkg
	rm -rf paper/.arxivcheck && cp -r paper/arxiv paper/.arxivcheck
	cd paper/.arxivcheck && for i in 1 2 3; do \
	    pdflatex -interaction=nonstopmode -halt-on-error paper-arxiv.tex >/dev/null || exit 1; done
	@! grep -q 'undefined' paper/.arxivcheck/paper-arxiv.log || \
	    { echo "FAIL: undefined references or citations"; exit 1; }
	@grep 'Output written' paper/.arxivcheck/paper-arxiv.log
	rm -rf paper/.arxivcheck

slides:
	cd presentation && pdflatex -interaction=nonstopmode presentation.tex && pdflatex -interaction=nonstopmode presentation.tex

clean:
	cd paper && rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.toc paper.pdf

mount_data:
	@if mountpoint -q $(DATA_DIR); then echo "$(DATA_DIR) already mounted"; exit 0; fi
	sudo mkfs.ext4 -F /dev/nvme1n1
	sudo mkdir -p $(DATA_DIR)
	sudo mount /dev/nvme1n1 $(DATA_DIR)
	sudo chown $$(id -u):$$(id -g) $(DATA_DIR)
	grep -q '/dev/nvme1n1' /etc/fstab || echo '/dev/nvme1n1 $(DATA_DIR) ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
