.PHONY: all test clean install figures

install:
	pip install -r requirements.txt

all: figures

figures:
	bash run_all.sh

test:
	python3 -m pytest tests/ -v

clean:
	rm -f figures/*.png results/*.json results/*.csv
