# make_pipeline_callgraph.py
import sys
from types import FrameType
from collections import defaultdict

# --- point to your entry ---
from main_emotionRecognition import run_pipeline_multi

JSON_PATH = r"..\experiment_data\session_data_log_2025-08-22.json"

# -------- which functions to show (fully qualified) --------
WANTED = {
    "main_emotionRecognition.run_pipeline",
    "parsing_script.parse_json_to_dataframe",
    "convert_timeStamp.convert_timestamps_safely",
    "label_density.label_session_density",
    "preprocess_vr_data.preprocess_and_tag_groups",
    "preprocess_vr_data.chunk_grouped_data",
    "token_embedder.build_categorical_vocabs",
    "token_embedder.TokenEmbedder.__init__",  # keep if you want module init
    "attention_runner.get_chunk_batch",
    "attention_runner.run_attention",
    "classifier_runner.run_classifier",
    "train_classifier.train_transformer_classifier",
}

LABELS = {  # prettier box labels
    "main_emotionRecognition.run_pipeline": "run_pipeline",
    "parsing_script.parse_json_to_dataframe": "parse_json_to_dataframe",
    "convert_timeStamp.convert_timestamps_safely": "convert_timestamps_safely",
    "label_density.label_session_density": "label_session_density",
    "preprocess_vr_data.preprocess_and_tag_groups": "preprocess_and_tag_groups",
    "preprocess_vr_data.chunk_grouped_data": "chunk_grouped_data",
    "token_embedder.build_categorical_vocabs": "build_categorical_vocabs",
    "token_embedder.TokenEmbedder.__init__": "TokenEmbedder.__init__",
    "attention_runner.get_chunk_batch": "get_chunk_batch",
    "attention_runner.run_attention": "run_attention",
    "classifier_runner.run_classifier": "run_classifier",
    "train_classifier.train_transformer_classifier": "train_transformer_classifier",
}

# -------- tracer that skips wrappers and links to nearest wanted ancestor --------
stack: list[str] = []
nodes: set[str] = set()
edges: defaultdict[tuple[str, str], int] = defaultdict(int)

def qname(frame: FrameType) -> str:
    mod = frame.f_globals.get("__name__", "?")
    name = frame.f_code.co_name
    if "self" in frame.f_locals:
        cls = type(frame.f_locals["self"]).__name__
        return f"{mod}.{cls}.{name}"
    return f"{mod}.{name}"

def nearest_wanted_ancestor() -> str | None:
    # search from top-1 down to bottom for first item in WANTED
    for qn in reversed(stack[:-1]):
        if qn in WANTED:
            return qn
    return None

def tracer(frame: FrameType, event: str, arg):
    if event == "call":
        callee = qname(frame)
        stack.append(callee)

        if callee in WANTED:
            nodes.add(callee)
            parent = nearest_wanted_ancestor()
            if parent:
                edges[(parent, callee)] += 1

    elif event == "return":
        if stack:
            stack.pop()
    return tracer

def write_dot(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        f.write('  rankdir=TB; node [shape=box, style=filled, fillcolor="#e6f2ff"];\n')
        for n in sorted(nodes):
            label = LABELS.get(n, n)
            f.write(f'  "{n}" [label="{label}"];\n')
        for (u, v), w in edges.items():
            f.write(f'  "{u}" -> "{v}" [label="{w}"];\n')
        f.write("}\n")

def main():
    sys.setprofile(tracer)
    try:
        run_pipeline_multi()
    finally:
        sys.setprofile(None)
        write_dot("pipeline_callgraph.dot")
        print("Wrote pipeline_callgraph.dot")

if __name__ == "__main__":
    main()
