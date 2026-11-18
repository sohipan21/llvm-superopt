# Entry points for the whole project. Toolchain targets need LLVM_BIN, LLVM_DIR, and
# ALIVE_TV in the environment (source the toolchain.env written by
# scripts/build_toolchain.sh). `make lint test` needs only uv.

# Install the package as a copy instead of an editable link. An editable install relies on
# a .pth file in the virtual environment, and on macOS such files can be flagged hidden
# (for example in folders synced by iCloud), in which case Python ignores them and the
# package cannot be imported.
export UV_NO_EDITABLE := 1

UV_RUN := uv run
PLUGIN_BUILD_DIR := build/harvest

.PHONY: all doctor plugin corpus harvest synth verify report test test-toolchain lint format clean-data

all: plugin corpus harvest synth verify report

plugin:
	cmake -G Ninja -S harvest -B $(PLUGIN_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR="$(LLVM_DIR)"
	ninja -C $(PLUGIN_BUILD_DIR)

doctor: plugin
	$(UV_RUN) superopt doctor

test:
	$(UV_RUN) pytest -m "not toolchain"

test-toolchain: plugin
	$(UV_RUN) pytest -m toolchain

lint:
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

format:
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .

clean-data:
	rm -rf data
