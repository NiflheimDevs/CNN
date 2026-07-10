"""
nngraph/onnx_export.py

Phase 9 stretch goal from plan.txt, now built: "ONNX export ... can be
added as sibling backends consuming the same validated AST rather
than forking the pipeline." This module is exactly that -- a second
backend next to codegen.py, not a replacement for it.

Deliberately does NOT reimplement graph traversal, topological
ordering, or forward()-body construction. codegen.generate() already
owns 100% of "how does this Program become a working PyTorch model" --
duplicating that logic here would create a second place every new
layer type or structural op has to be taught, and the two would drift
out of sync exactly the way layer_catalogue.py's own docstring warns
about for PYTORCH_CLASS_MAP/EXPECTED_PARAM_TYPES.

Instead: reuse codegen.generate(program) to get real, working Python
source; exec() that source in a private namespace to obtain the
nn.Module *class*; instantiate it; feed it a dummy input shaped from
Model.input/Config; and hand the live model + tensor to
torch.onnx.export(). ONNX export needs an actual traced/scripted
module -- there's no "generate ONNX text" shortcut the way DOT or
Python source are just string templates, so this backend's job is
orchestration, not generation.

Precondition (same contract as codegen.py): `program` has already
passed SemanticAnalyzer.analyze() with zero ERROR diagnostics. This
module does not re-validate the graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nngraph.ast_nodes import Program
from nngraph.codegen import CodegenError, generate


class OnnxExportError(Exception):
    """Raised for anything that stops export -- missing torch/onnx
    dependencies, a generated module that doesn't expose the expected
    nn.Module class, or torch.onnx.export() itself failing. Codegen
    failures are NOT wrapped in this type; they surface as the
    CodegenError they already are, same as compiler.py's compile
    mode, so callers only need one except clause per concern."""


def _safe_identifier(name: str) -> str:
    """Mirrors codegen._safe_identifier exactly (Python-keyword-as-id
    escaping). Not imported directly because it's a private, leading-
    underscore helper of codegen.py -- copying this one small pure
    function keeps onnx_export.py from reaching into codegen's
    internals, while everything with actual graph semantics (the
    class body itself) still comes from calling codegen.generate().
    """
    import keyword
    return f"{name}_" if keyword.iskeyword(name) else name


def _load_model_class(source: str, class_name: str) -> Any:
    """exec()'s codegen's generated source in an isolated namespace
    and returns the nn.Module subclass it defines.

    Using exec() instead of writing a temp .py file and importing it
    avoids leaving stray files/sys.path entries behind and keeps this
    a pure in-memory pipeline stage, matching how the rest of the
    compiler passes a Program object from phase to phase without
    touching disk until the final write.

    The generated source's own `if __name__ == '__main__':` block
    (see codegen._emit_main_block) never runs here -- exec()'s
    __name__ is "onnx_export_module", not "__main__" -- so this never
    accidentally allocates a device tensor or prints a shape as a
    side effect of loading the class.
    """
    try:
        import torch  # noqa: F401  -- imported for its side effect: makes
        # `torch` / `torch.nn` resolvable when the generated source's own
        # `import torch` / `import torch.nn as nn` lines execute below.
    except ImportError as e:
        raise OnnxExportError(
            "ONNX export requires PyTorch, which is not installed in this "
            "environment. Install it with `pip install torch` (see "
            "https://pytorch.org/get-started/locally/ for the right build "
            "for your platform/CUDA version), then re-run with --onnx."
        ) from e

    namespace: dict[str, Any] = {"__name__": "onnx_export_module"}
    try:
        exec(compile(source, "<nngraph-generated>", "exec"), namespace)
    except Exception as e:
        raise OnnxExportError(
            f"the code generated for model '{class_name}' failed to "
            f"execute while loading it for ONNX export: {e}"
        ) from e

    model_class = namespace.get(class_name)
    if model_class is None or not (
        isinstance(model_class, type) and issubclass(model_class, torch.nn.Module)
    ):
        raise OnnxExportError(
            f"expected generated source to define a class named "
            f"'{class_name}' inheriting from nn.Module, but it did not. "
            f"This indicates codegen.py's class-naming convention "
            f"(_safe_identifier(program.model.name)) has drifted from "
            f"this module's assumption -- both should stay in lockstep."
        )
    return model_class


def export_onnx(
    program: Program,
    output_path: str,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> None:
    """Compiles `program` to a live PyTorch model (via codegen.generate
    + exec, see _load_model_class) and exports it to `output_path` as
    an ONNX graph.

    What ONNX export IS: a static computation-graph dump (op nodes +
    tensor shapes/dtypes + weights) that runtimes other than PyTorch
    -- ONNX Runtime, TensorRT, CoreML, mobile/edge runtimes, etc. --
    can load and run, without needing Python or PyTorch installed on
    the machine that eventually serves the model. It's produced by
    tracing one real forward() call with a dummy input of the right
    shape and recording every tensor op that ran.

    batch_size for the dummy input comes from config { batch_size =
    ... } (the same field codegen's __main__ scaffold already uses),
    defaulting to 1 if there's no config block. dynamic_batch=True
    (the default) marks that dimension as symbolic in the exported
    graph, so the ONNX model isn't hard-locked to whatever batch size
    happened to be used for tracing -- callers can feed it batches of
    any size at inference time, which is what you want for a model
    artifact instead of a one-off trace.

    Export always runs on CPU regardless of config { device = ... }:
    the exported graph is a device-independent computation graph, and
    requiring CUDA to be available just to run `--onnx` would make
    this backend fail in plenty of environments (like this one) that
    have no GPU but every reason to want an .onnx file.
    """
    class_name = _safe_identifier(program.model.name)
    input_name = _safe_identifier(program.model.input.name)
    output_name = _safe_identifier(program.model.output)

    try:
        source = generate(program)
    except CodegenError:
        raise  # let compiler.py's existing CodegenError handling own this

    model_class = _load_model_class(source, class_name)

    import torch  # safe: _load_model_class already proved this import works

    model = model_class()
    model.to("cpu")
    model.eval()

    batch_size = 1
    if program.config is not None:
        configured = program.config.entries.get("batch_size")
        if isinstance(configured, int) and configured > 0:
            batch_size = configured

    dummy_input = torch.randn(batch_size, *program.model.input.shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs: dict[str, Any] = dict(
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    # torch >= 2.5 defaults to the newer "dynamo" exporter, which needs
    # the separate `onnxscript` package installed to serialize the
    # graph and doesn't accept dynamic_axes= the way the classic
    # TorchScript-based exporter does. This backend targets the
    # dynamic_axes API and has no other reason to need onnxscript as a
    # dependency, so pin dynamo=False when that parameter exists.
    # Older torch versions (< 2.5) don't have a `dynamo` parameter at
    # all -- their export() is always the classic path -- so passing
    # it there would be a TypeError, hence the signature check.
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    try:
        torch.onnx.export(model, dummy_input, str(out), **export_kwargs)
    except Exception as e:
        raise OnnxExportError(f"torch.onnx.export() failed: {e}") from e
