.PHONY: build run stop clean test

build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

clean:
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -r {} +

test:
	python -m pytest tests/

setup:
	chmod +x setup.sh
	./setup.sh