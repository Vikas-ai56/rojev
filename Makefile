# Each target is one command; read it here before running. Use a separate terminal for each long-running one.
export PHOENIX_WORKING_DIR := $(CURDIR)/data/phoenix
export PHOENIX_HOST := 127.0.0.1
GGUF ?= mradermacher/OneJev1.5-0.8B-GGUF:Q8_0

.PHONY: phoenix qev sim sim-viewer test

phoenix:      ## trace UI + collector at http://localhost:6006 (traces stored in data/phoenix)
	uv run phoenix serve

qev:          ## OneJev selector at http://127.0.0.1:8000 (loopback only)
	uv run qev serve --gguf $(GGUF) --host 127.0.0.1 --port 8000

sim:          ## scripted trajectory, headless; add ARGS="--onejev --hide-box" etc.
	uv run python examples/scripted_trajectory.py $(ARGS)

sim-viewer:   ## same, in the MuJoCo viewer
	uv run mjpython examples/scripted_trajectory.py --viewer $(ARGS)

test:         ## G0 install gate + SDK tests
	uv run pytest -v
