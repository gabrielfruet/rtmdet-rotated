
# RTMDet-Rotated

A clean, type-safe PyTorch implementation of RTMDet-Rotated with a modular, hackable design.

## Why this project

If you have tried MMRotate or MMDetection, you know the setup can be a hassle: heavy dependencies,
large configs, and a lot of framework wiring just to test an idea. This repo keeps the core parts
small and readable so you can move fast, understand the whole pipeline, and change what matters.

"You only really understand something when you can make it simple."

## What you get

- Minimal, explicit modules (ops, assigners, loss, model)
- Type hints with tensor shape annotations (jaxtyping)
- Tests for core geometry and assignment logic

## Quick start

```bash
uv sync
pytest
```

## Project layout

```
src/rtmdet/
	ops.py        # geometry + point ops
	assigner.py   # assignment logic
	loss.py       # loss functions
	model.py      # model components
tests/
```
