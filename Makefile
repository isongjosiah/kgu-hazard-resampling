.PHONY: setup dev check fmt lint test test-rust test-py clean pilots

setup:            ## create the environment and build the Rust engine
	uv sync

dev:              ## rebuild the Rust engine after changing it
	uv run maturin develop --uv

fmt:              ## format everything
	cargo fmt --all
	uv run ruff format .
	uv run ruff check --fix .

lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings
	uv run ruff format --check .
	uv run ruff check .

test-rust:
	cargo test -p hazres-grid

test-py:
	uv run pytest

test: test-rust test-py

check: lint test  ## the gate: must pass before every merge

clean:
	cargo clean
	rm -rf .venv .pytest_cache .ruff_cache

PILOTS = pilot_sicily pilot_trentino pilot_se_spain

pilots:           ## cache and run all three pilot regions, then summarise
	@for p in $(PILOTS); do \
		uv run hazres data cache --experiment configs/experiments/$$p.yaml && \
		uv run hazres run --experiment configs/experiments/$$p.yaml || exit 1; \
	done
	uv run hazres summary
