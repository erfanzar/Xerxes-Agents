# Native Bun documentation build entrypoint.

BUN ?= bun
RUNTIME_DIRECTORY ?= ../xerxes

.PHONY: help html

help:
	@echo "Use 'make html' to build documentation with Bun."

html:
	@$(BUN) run --cwd $(RUNTIME_DIRECTORY) docs:build
