.PHONY: setup dev check fmt lint test test-rust test-py clean

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
