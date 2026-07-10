# NNGraph

NNGraph is a small domain-specific language (DSL) for describing neural
network architectures as explicit node/edge graphs, plus a compiler that
turns those descriptions into ready-to-run PyTorch code.

Instead of hand-writing an `nn.Module` and wiring up `forward()` yourself,
you describe the model as nodes (layers) and edges (data flow) in a
`.nng` file. The compiler parses it, validates it (shapes, cycles,
unreachable nodes, parameter types, ...), and generates a matching PyTorch
class — or renders the graph as a diagram, or exports it straight to ONNX.

```
model MLP {
  input x : tensor(784)
  output out
}

graph {
  node fc1   : Linear(in_features=784, out_features=256)
  node relu1 : ReLU()
  node drop1 : Dropout(p=0.3)
  node fc2   : Linear(in_features=256, out_features=128)
  node relu2 : ReLU()
  node fc3   : Linear(in_features=128, out_features=10)
  node out   : Softmax(dim=1)

  edge x     -> fc1
  edge fc1   -> relu1
  edge relu1 -> drop1
  edge drop1 -> fc2
  edge fc2   -> relu2
  edge relu2 -> fc3
  edge fc3   -> out
}

config {
  batch_size = 64
  device = "cuda"
}
```

```bash
python compiler.py mlp.nng --output mlp.py
```

produces a standard `nn.Module` subclass with `__init__` and `forward`
already wired up.

## Features

- **Graph-based model definition** — layers are `node`s, data flow is
  `edge`s. Branching, merging, and residual/skip connections are first
  class, not bolted on.
- **Static semantic analysis** — before any code is generated, the
  compiler checks for duplicate node ids, undefined edge references,
  orphan nodes, cycles, unreachable input/output, wrong arity on
  `Residual()` nodes, and per-layer parameter types/required arguments.
  All problems are collected and reported together, not one at a time.
- **Shape inference** — layers that only need to know how many
  features/channels are flowing *in* (`in_features`, `in_ch`,
  `num_features`, `normalized_shape`, ...) can omit that parameter; the
  compiler infers it from the upstream graph shape where possible.
- **PyTorch code generation** — deterministic, topologically-sorted
  `forward()` bodies; stateless ops (`Add`, `Concat`, `Residual`,
  `Split`) are inlined as tensor expressions instead of being registered
  as submodules.
- **Graph visualization** — render the node graph as a Graphviz diagram
  directly from the parsed source, even before it's semantically valid,
  to help debug wiring issues visually.
- **ONNX export** — trace the generated model and export it directly to
  an `.onnx` file, with optional dynamic batch dimension.

## How it works

```
.nng source
    │  ANTLR4 lexer/parser
    ▼
parse tree
    │  ASTBuilder
    ▼
AST (Program / Model / Node / Edge / Config)
    │  SemanticAnalyzer  (symbol table, cycle/reachability checks,
    │                      shape inference, parameter validation)
    ▼
validated AST ──► codegen.py        ──► generated PyTorch source (.py)
              ──► dot_visualizer.py ──► Graphviz diagram (.png/.svg/...)
              ──► onnx_export.py    ──► traced ONNX graph (.onnx)
```

Diagnostics (syntax and semantic errors/warnings) are collected into a
single list and reported together, sorted by line number. Code
generation only runs once there are zero `ERROR`-level diagnostics;
warnings (e.g. orphan nodes) don't block compilation.

## Installation

Requires **Python 3.10+**.

```bash
git clone git@github.com:NiflheimDevs/CNN.git
cd CNN
pip install -r requirements.txt
```

The grammar's parser/lexer are already generated and checked in under
`gen/`. If you ever change `NNGraph.g4`, regenerate them with the
[ANTLR 4.13.2](https://www.antlr.org/) tool:

```bash
java -jar antlr-4.13.1-complete.jar -Dlanguage=Python3 -visitor -o gen/ NNGraph.g4
```

Optional dependencies, only needed for specific modes:

| Feature | Requirement |
|---|---|
| `--onnx` export | `torch` (`pip install torch`) |
| `--dot` image rendering | [Graphviz](https://graphviz.org/download/) installed and on `PATH` (the `.dot` source itself can still be generated without it) |

## Usage

```
python compiler.py <input.nng> [options]
```

### Compile to PyTorch (default)

```bash
python compiler.py mlp.nng --output mlp.py
```

Runs the full pipeline: parse → build AST → semantic analysis → codegen.
Stops and prints diagnostics if there are any syntax or semantic errors.

### Visualize the graph

```bash
python compiler.py mlp.nng --dot --output mlp.png
```

Renders the node graph with Graphviz. This only checks *syntax*, not
semantics, so you can visualize a graph that doesn't compile yet (has
undefined edges, cycles, etc.) to see what's wrong.

```bash
python compiler.py mlp.nng --dot --dot-only --output mlp.dot   # source only
python compiler.py mlp.nng --dot --format svg --output mlp.svg # other formats
```

### Export to ONNX

```bash
python compiler.py mlp.nng --onnx --output mlp.onnx
```

Runs the same two gates as compile mode, then loads the generated model
in memory and traces a real `forward()` pass with `torch.onnx.export()`.
Requires PyTorch.

```bash
python compiler.py mlp.nng --onnx --opset 18 --output mlp.onnx
python compiler.py mlp.nng --onnx --no-dynamic-batch --output mlp.onnx
```

## Language guide

A `.nng` file has three parts: a `model` declaration, a `graph` block,
and an optional `config` block.

```
model <Name> {
  input <id> : tensor(<dim>, <dim>, ...)
  output <id>
}

graph {
  node <id> : <LayerType>(<param>=<value>, ...)
  edge <id> -> <id> [label="<name>"]
}

config {
  <key> = <value>
}
```

- **Scalars** are bare literals: `in_features=784`, `p=0.3`,
  `device="cuda"`.
- **Shapes** are parenthesized: `tensor(3, 224, 224)`,
  `normalized_shape=(128)`.
- Edge labels are required to disambiguate the two inputs of a
  `Residual()` node — the edge labeled `"shortcut"` is treated as the
  identity/skip branch, the other as the main path.
- Nodes whose output feeds more than one downstream node are
  automatically bound to a named variable in the generated `forward()`;
  everything else is threaded through by node id.

See [`NNGraph_DSL_Specification.pdf`](./NNGraph_DSL_Specification.pdf)
for the full language spec, and `mlp.nng`, `mlp2.nng`, `mlp3.nng`, and
`mlp4.nng` for worked examples covering a plain MLP, a residual block,
an Inception-style branch/merge block, and a Transformer encoder layer.

### Supported layer types

| Category | Types |
|---|---|
| Layers | `Linear`, `Conv1d`, `Conv2d`, `BatchNorm2d`, `LayerNorm`, `MaxPool2d`, `AvgPool2d`, `Dropout`, `Flatten`, `Embedding`, `MultiHeadAttn`, `LSTM`, `GRU` |
| Activations | `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax`, `LeakyReLU`, `ELU` |
| Structural ops (no backing `nn.Module`) | `Add`, `Concat`, `Residual`, `Split` |

## Project structure

```
CNN/
├── compiler.py                     # CLI entry point (compile / --dot / --onnx)
├── NNGraph.g4                      # ANTLR4 grammar
├── NNGraph_DSL_Specification.pdf   # Full language specification
├── plan.txt                        # Implementation plan / design notes
├── TODO.md                         # Known gaps and stretch goals
├── mlp*.nng                        # Example DSL source files
├── gen/                            # ANTLR-generated lexer/parser/visitor
└── nngraph/
    ├── ast_nodes.py                # AST dataclasses (Program, Model, Node, Edge, Config)
    ├── ast_builder.py              # Parse tree -> AST
    ├── semantic_analyzer.py        # Symbol table, validation, shape inference
    ├── shape_inference.py          # Shape propagation for INFERABLE_PARAM
    ├── shape_interface.py          # Shared shape-related type/interface definitions
    ├── layer_catalogue.py          # Shared layer-type constants (structural ops, inferable params)
    ├── codegen.py                  # AST -> PyTorch source
    ├── dot_visualizer.py           # AST -> Graphviz DOT / rendered image
    ├── onnx_export.py              # Generated model -> traced .onnx file
    ├── graph_utils.py              # Topological sort, cycle detection, reachability
    ├── diagnostics.py              # Diagnostic type + formatting/sorting
    └── error_listener.py           # ANTLR error listener feeding into Diagnostics
```

## Known limitations / roadmap

See [`TODO.md`](./TODO.md) for the current list. Notably:

- Semantic analysis is not yet optimized for large graphs.
- Duplicate diagnostics can appear for a node that is both undefined
  *and* unreachable.
- Generated `forward()` bodies bind more intermediate variables than
  strictly necessary.
- No automated test suite yet (planned: golden-file tests comparing
  `ast.dump()` output for the bundled examples, plus one negative
  fixture per semantic rule).

## License

No license has been chosen for this project yet — add one (e.g. MIT,
Apache-2.0) before treating it as open source.
