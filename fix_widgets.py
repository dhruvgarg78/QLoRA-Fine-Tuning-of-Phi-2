import json, pathlib

def fix(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    md = nb.get("metadata", {})
    # If there's a widgets block and it's malformed, drop it.
    w = md.get("widgets")
    drop = False
    if isinstance(w, dict):
        mime = "application/vnd.jupyter.widget-state+json"
        if mime in w:
            drop = not isinstance(w[mime], dict) or "state" not in w[mime]
        else:
            drop = True
    if drop:
        md.pop("widgets", None)
        nb["metadata"] = md
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("fixed:", path)
    else:
        print("ok    :", path)

for p in pathlib.Path(".").rglob("*.ipynb"):
    fix(p)
