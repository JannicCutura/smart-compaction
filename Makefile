.PHONY: install_tex install_py tex clean params generate mount_data snapshot restore extract compact label train evaluate pipeline ablation ablation_plots compact_512 query_bench

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

clean:
	cd paper && rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.toc paper.pdf

mount_data:
	@if mountpoint -q $(DATA_DIR); then echo "$(DATA_DIR) already mounted"; exit 0; fi
	sudo mkfs.ext4 -F /dev/nvme1n1
	sudo mkdir -p $(DATA_DIR)
	sudo mount /dev/nvme1n1 $(DATA_DIR)
	sudo chown $$(id -u):$$(id -g) $(DATA_DIR)
	grep -q '/dev/nvme1n1' /etc/fstab || echo '/dev/nvme1n1 $(DATA_DIR) ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
