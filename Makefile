.PHONY: all audit check clean doc lint test

MAKEFILE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: audit check doc lint test

audit:
	cargo deny --log-level=error check -s

check:
	cargo fmt -q --all --check
	cargo clippy -q --all-targets --all-features -- -D warnings

clean:
	@rm -rf "$(MAKEFILE_DIR)/target"

doc: FLAGS :=
doc:
	cargo doc -q --workspace --no-deps --all-features $(FLAGS)

lint:
	cargo hack -q --feature-powerset check
	cargo +nightly udeps -q --all-features
	cargo outdated -q -R --exit-code=1

test:
	cargo llvm-cov --workspace --all-features --all-targets
