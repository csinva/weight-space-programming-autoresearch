"""Bounded search API. Session identity belongs to the server, never the request."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socketserver
import tempfile
import threading


MAX_REQUEST = 1_100_000
MAX_SOURCE = 1_000_000
MAX_LIST_ROWS = 20
MAX_LIST_BYTES = 12_000
COMPACT_FIELDS = ("id", "idea", "proposer", "status", "search_score", "metric", "region", "parent_id",
                  "parent_score_delta", "n_params", "seconds", "dataset", "subregion_scores",
                  "error_reason", "candidate_error")
PUBLIC_FIELDS = ("id", "idea", "experiment_notes", "proposer", "dataset", "status",
                 "test_corr", "roi_corr", "n_params", "model_loc", "seconds", "created_at",
                 "metric", "region", "stage", "protocol_version", "subject_scores", "rsa_diagnostics",
                 "parent_id", "model_sha256", "repair_depth")


def _decoded(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _selected_details(value, allowed, required=None):
    """Filter ROI maps at every depth, including nested source-group subjects."""
    if isinstance(value, list):
        return [_selected_details(item, allowed, required) for item in value]
    if not isinstance(value, dict):
        return value
    from neuro_eval.benchmarks import SEARCH_REGIONS, SEARCH_SUBREGIONS
    known = set(SEARCH_REGIONS.values()) | {name for streams in SEARCH_SUBREGIONS.values()
                                          for names in streams.values() for name in names}
    forbidden = known - allowed
    result = {}
    for key, item in value.items():
        if key == "search_region" and item in SEARCH_REGIONS and SEARCH_REGIONS[item] not in allowed:
            continue
        if key in forbidden or (isinstance(item, str) and item in forbidden):
            continue
        if key in {"regions", "roi_corr", "subregions", "unavailable_regions"}:
            result[key] = {name: _selected_details(entry, allowed, required)
                           for name, entry in item.items() if name in allowed} if isinstance(item, dict) else {}
        elif key == "subject_roi_scores" and isinstance(item, dict):
            result[key] = {subject: {name: entry for name, entry in entries.items() if name in allowed}
                           for subject, entries in item.items() if isinstance(entries, dict)}
        else:
            result[key] = _selected_details(item, allowed, required)
    # Legacy group totals could average both streams; use the selected pooled ROI.
    if "rsa" in result and isinstance(result.get("regions"), dict):
        pooled = next((entry for name, entry in result["regions"].items()
                       if name.endswith("visual stream")), None)
        result["rsa"] = pooled.get("rsa") if isinstance(pooled, dict) else pooled
        if result["rsa"] is None:
            result["status"] = "unavailable"
    if required and "n_subjects" in result and (result["n_subjects"] != len(required)
            or ("subjects" in result and set(result["subjects"]) != required)):
        result["status"] = "unavailable"
        for key in ("score", "rsa", "test"):
            if key in result:
                result[key] = None
    return result


def _candidate_trace(value):
    """Bound image-only stderr and redact absolute paths before presentation."""
    value = value[-1500:]
    value = re.sub(r"(?<![\w])/(?:[^\s\"\'<>:,;()]+)", "[path]", value)
    return value


def public_row(row, *, details=False):
    if row is None:
        return None
    result = {key: row[key] for key in PUBLIC_FIELDS if key in row}
    result["search_score"] = row.get("test_corr")
    if row.get("metric") == "rsa":
        result["search_rsa"] = row.get("test_corr")
    metadata = row.get("evaluation_metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = None
    if isinstance(metadata, dict) and isinstance(metadata.get("rsa_diagnostics"), dict):
        result["rsa_diagnostics"] = metadata["rsa_diagnostics"]
    # Only trusted evaluator metadata may mark stderr as originating inside the
    # image-only candidate worker. Never expose raw scorer status/tracebacks.
    if details and isinstance(metadata, dict) and isinstance(metadata.get("candidate_error"), str):
        result["candidate_error"] = _candidate_trace(metadata["candidate_error"])
    if row.get("status") == "error":
        message = "\n".join(str(metadata.get(key, "")) for key in ("status", "candidate_error")) if isinstance(metadata, dict) else ""
        result["error_reason"] = "Evaluation failed; inspect candidate shape, resources and model contract"
        for needles, reason in (
            (("cuda out of memory", "cuda error: out of memory", "cuda memory", "outofmemoryerror", "oom on device"),
             "CUDA out of memory; reduce internal batch size or activation memory"),
            (("timeout", "timed out"), "Evaluation timed out; reduce model runtime"),
            (("parameter", "n_params"), "Parameter contract failed; check n_params and the parameter limit"),
            (("nonfinite", "non-finite", "nan", "infinite"), "Nonfinite model features or score; return finite features"),
            (("constant",), "Degenerate constant features or score"),
            (("batch", "per-image"), "Per-image independence check failed; check batch dependence, mutable state and numerical stability"),
            (("shape", "width", "dimension"), "Feature shape or width contract failed; return (N, D) with D < 10000"),
            (("source identity", "hash"), "Archived model identity check failed"),
            (("cohort", "incompatible evaluation"), "Scientific protocol validation failed"),
            (("isolat", "namespace", "bubblewrap"), "Isolated candidate execution failed"),
        ):
            if any(re.search(r"\bnan\b", message, re.IGNORECASE) if word == "nan"
                   else word in message.lower() for word in needles):
                result["error_reason"] = reason
                break
    from neuro_eval.benchmarks import SEARCH_REGIONS, search_subregions, protocol_for
    names = search_subregions(row.get("dataset", "laion"), row.get("region", "ventral"))
    pooled = SEARCH_REGIONS[row.get("region", "ventral")]
    allowed = {pooled, *names}
    stream = metadata.get("stream_diagnostics", {}) if isinstance(metadata, dict) else {}
    subregions = stream.get("subregions", {}) if isinstance(stream, dict) else {}
    if not isinstance(subregions, dict):
        subregions = {}
    required = set(protocol_for(row.get("dataset", "laion"), row.get("region", "ventral"),
                                row.get("metric", "rsa")).search_subjects)
    result["subregion_scores"] = {}
    for name in names:
        entry = subregions.get(name, {})
        valid = isinstance(entry, dict) and entry.get("status") == "ok" and entry.get("score") is not None
        valid = valid and set(entry.get("subjects", [])) == required and entry.get("n_subjects") == len(required)
        result["subregion_scores"][name] = {"status": "ok" if valid else "unavailable",
                                           "score": entry.get("score") if valid else None}
    if details:
        for key in ("roi_corr", "subject_scores", "rsa_diagnostics", "experiment_notes"):
            parsed = _decoded(result.get(key))
            if parsed is not None:
                result[key] = parsed
        if isinstance(result.get("roi_corr"), dict):
            result["roi_corr"] = {key: val for key, val in result["roi_corr"].items() if key in allowed}
        result = _selected_details(result, allowed, required)
        if stream:
            result["stream_diagnostics"] = _selected_details(stream, allowed, required)
    else:
        result = {key: value for key, value in result.items() if key in COMPACT_FIELDS}
        if isinstance(result.get("idea"), str):
            result["idea"] = result["idea"][:240]
        if row.get("status") == "error":
            result.pop("subregion_scores", None)

    return result


class Broker:
    """One private socket per session; caller must mount only that socket's directory."""

    def __init__(self, run_dir, role, proposer, socket_path):
        global queue
        from search import queue
        if role not in {"proposer", "coordinator"}:
            raise ValueError("Only proposer and coordinator sessions have broker access")
        if role in {"proposer"} and not re.fullmatch(r"p[0-9]+", proposer or ""):
            raise ValueError("Invalid proposer identity")
        self.run = str(Path(run_dir).resolve())
        self.contract = queue.search_contract(self.run)
        self.role, self.proposer = role, proposer
        self.socket_path = str(socket_path)
        self.server = None
        self.thread = None

    def _rows(self, sql, args=()):
        with queue.connect(self.run) as con:
            # Contract fields are canonical validated enums, never caller input.
            con.execute("CREATE TEMP VIEW run_models AS SELECT * FROM main.models "
                        f"WHERE dataset='{self.contract['NEURO_DATASET']}' "
                        f"AND region='{self.contract['NEURO_REGION']}' "
                        f"AND metric='{self.contract['NEURO_METRIC']}' AND stage='search'")
            return [dict(row) for row in con.execute(sql.replace("FROM models", "FROM run_models"), args)]


    def _public(self, row, *, details=False, compact=False):
        result = public_row(row, details=details)
        if row and row.get("parent_id") and row.get("test_corr") is not None:
            parents = self._rows("SELECT * FROM models WHERE id=? "
                                 "AND stage='search' AND status='ok'", (row["parent_id"],))
            if parents and parents[0]["test_corr"] is not None:
                result["parent_score_delta"] = row["test_corr"] - parents[0]["test_corr"]
                parent = public_row(parents[0])
                for name, entry in result.get("subregion_scores", {}).items():
                    previous = parent.get("subregion_scores", {}).get(name, {})
                    if entry.get("score") is not None and previous.get("score") is not None:
                        entry["parent_score_delta"] = entry["score"] - previous["score"]
        if compact and result is not None:
            result = {key: result[key] for key in COMPACT_FIELDS if key in result}
            if isinstance(result.get("idea"), str):
                result["idea"] = result["idea"][:240]
        return result

    def _island(self, name):
        if not isinstance(name, str) or not re.fullmatch(r"p[0-9]+", name):
            raise ValueError("Invalid island")
        path = Path(self.run) / "proposers" / name
        if not path.is_dir() or path.is_symlink():
            raise ValueError("Unknown island")
        return path

    def _own_row(self, mid):
        if not isinstance(mid, str) or not re.fullmatch(r"[a-z0-9]{5}", mid):
            raise ValueError("Invalid model ID")
        rows = self._rows("SELECT * FROM models WHERE id=?", (mid,))
        if not rows or (self.role in {"proposer"} and os.environ.get("VANILLA_AUTORESEARCH") == "1"
                        and rows[0]["proposer"] != self.proposer):
            raise PermissionError("Model is not available to this session")
        return rows[0]

    def _source(self, row):
        # Derive archive path from validated ID; never trust a DB path or a caller path.
        mid = row["id"]
        if not re.fullmatch(r"[a-z0-9]{5}", mid):
            raise ValueError("Invalid archive identity")
        path = Path(self.run) / "models" / (mid + ".py")
        if path.is_symlink():
            raise PermissionError("Invalid archive")
        with path.open() as source:
            text = source.read(MAX_SOURCE + 1)
        if len(text) > MAX_SOURCE:
            raise ValueError("Archive exceeds API limit")
        return text

    def dispatch(self, request):
        if not isinstance(request, dict):
            raise ValueError("Request must be an object")
        cmd = request.get("command")
        if request.get("details") and cmd != "status":
            raise ValueError("Detailed feedback is available through status for one model ID")
        common = {"history", "status", "source", "summary", "direction", "top-models"}
        proposer_commands = {"propose-model", "my-best", "random-models", "peek-island", "source"}
        coordinator_commands = {"top-models", "set-direction", "reset-island"}
        allowed = common | (proposer_commands if self.role in {"proposer"} else coordinator_commands)
        if cmd not in allowed:
            raise PermissionError("Operation is not available to this session")
        limit = request.get("limit", 10)
        if type(limit) is not int or not 1 <= limit <= MAX_LIST_ROWS:
            raise ValueError("Limit must be an integer between 1 and 20")
        if cmd == "propose-model":
            source, idea = request.get("source"), request.get("idea", "")
            if not isinstance(source, str) or not 0 < len(source.encode()) <= MAX_SOURCE:
                raise ValueError("Source must be nonempty and at most 1 MB")
            if not isinstance(idea, str) or len(idea) > 500:
                raise ValueError("Idea must be at most 500 characters")
            parent_id = request.get("parent_id")
            if parent_id is not None:
                self._own_row(parent_id)
            # Propose only parses source; it never imports or executes candidate code.
            with tempfile.TemporaryDirectory(prefix="neuro-submit-") as directory:
                path = Path(directory) / "model.py"
                path.write_text(source)
                return queue.propose(str(path), idea=idea, proposer=self.proposer, run=self.run,
                                   parent_id=parent_id, max_pending=1)
        if cmd in {"status", "source"}:
            row = self._own_row(request.get("id"))
            if cmd == "source":
                return {"id": row["id"], "code": self._source(row)}
            return self._public(row, details=request.get("details") is True)
        if cmd == "my-best":
            island = Path(self.run) / "state" / self.proposer
            island.mkdir(parents=True, exist_ok=True)
            marker = island / "reset_at"
            reset_at = marker.read_text().strip() if marker.exists() else ""
            (island / "reset_seen").write_text(reset_at)
            rows = self._rows("SELECT * FROM models WHERE proposer=? AND status='ok' "
                              "AND stage='search' AND created_at>? "
                              "ORDER BY test_corr DESC LIMIT 1", (self.proposer, reset_at))
            if not rows:
                return None
            return self._public(rows[0], compact=True)
        if cmd in {"history", "top-models", "random-models"}:
            if cmd == "random-models" and os.environ.get("VANILLA_AUTORESEARCH") == "1":
                raise PermissionError("Cross-island feedback is disabled")
            where = "stage='search'"
            args = []
            if self.role in {"proposer"} and os.environ.get("VANILLA_AUTORESEARCH") == "1":
                where += " AND proposer=?"
                args.append(self.proposer)
            if cmd != "history":
                where += " AND status='ok'"
            order = {"history": "created_at DESC", "top-models": "test_corr DESC",
                     "random-models": "RANDOM()"}[cmd]
            rows = self._rows(f"SELECT * FROM models WHERE {where} ORDER BY {order} LIMIT ?",
                              (*args, limit))
            result = []
            for row in rows:
                candidate = self._public(row, compact=True)
                if len(json.dumps([*result, candidate], indent=2).encode()) > MAX_LIST_BYTES:
                    break
                result.append(candidate)
            return result
        if cmd == "peek-island":
            if os.environ.get("VANILLA_AUTORESEARCH") == "1":
                raise PermissionError("Cross-island feedback is disabled")
            islands = self._rows("SELECT DISTINCT proposer FROM models WHERE proposer<>? "
                                 "AND status='ok' AND stage='search' "
                                 "ORDER BY RANDOM() LIMIT 1", (self.proposer,))
            if not islands:
                return None
            rows = self._rows("SELECT idea,experiment_notes FROM models WHERE proposer=? "
                              "AND status='ok' AND stage='search' "
                              "ORDER BY test_corr DESC LIMIT 1", (islands[0]["proposer"],))
            return {"idea": (rows[0].get("idea") or "")[:240],
                    "experiment_notes": (rows[0].get("experiment_notes") or "")[:4000]} if rows else None
        if cmd == "summary":
            where, args = "stage='search'", ()
            if self.role in {"proposer"} and os.environ.get("VANILLA_AUTORESEARCH") == "1":
                where += " AND proposer=?"
                args = (self.proposer,)
            return self._rows(f"SELECT proposer,status,COUNT(*) AS count,MAX(test_corr) AS best_score "
                              f"FROM models WHERE {where} GROUP BY proposer,status", args)
        if cmd == "direction":
            if self.role in {"proposer"}:
                islands = [self._island(self.proposer)]
            else:
                islands = [self._island(path.name) for path in
                           sorted((Path(self.run) / "proposers").glob("p[0-9]*"))]
            return {path.name: (Path(self.run) / "state" / path.name / "direction.txt").read_text()[:2000]
                    if (Path(self.run) / "state" / path.name / "direction.txt").exists() else "" for path in islands}
        island = self._island(request.get("proposer"))
        direction = request.get("direction", "")
        if not isinstance(direction, str) or not 0 < len(direction) <= 2000:
            raise ValueError("Direction must contain 1 to 2000 characters")
        if cmd == "set-direction":
            state = Path(self.run) / "state" / island.name
            state.mkdir(parents=True, exist_ok=True)
            (state / "direction.txt").write_text(direction.strip() + "\n")
            return {"proposer": island.name, "direction": direction.strip()}
        # queue.reset_island's environment check is not the authorization boundary;
        # this trusted session has already checked its immutable coordinator role.
        if os.environ.get("LAB_PROPOSER"):
            raise RuntimeError("Conductor broker must launch without LAB_PROPOSER")
        return queue.reset_island(island.name, role=direction, run=self.run)

    def start(self):
        broker = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(30)
                try:
                    line = self.rfile.readline(MAX_REQUEST + 1)
                    if len(line) > MAX_REQUEST or not line.endswith(b"\n"):
                        raise ValueError("Request exceeds limit or is incomplete")
                    result = broker.dispatch(json.loads(line))
                    payload = {"ok": True, "result": result}
                except (ValueError, PermissionError, KeyError) as exc:
                    message = "Search file operation failed" if isinstance(exc, OSError) and exc.filename else str(exc)[:2000]
                    payload = {"ok": False, "error": message}
                except Exception:
                    # Infrastructure exceptions may include host paths or credentials.
                    payload = {"ok": False, "error": "Search operation failed; consult operator logs"}
                self.wfile.write((json.dumps(payload, allow_nan=False) + "\n").encode())

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
            block_on_close = False

        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("Broker socket already exists")
        self.server = Server(str(path), Handler)
        os.chmod(path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            Path(self.socket_path).unlink(missing_ok=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()


import argparse
import json
import os
from pathlib import Path
import socket
import time

MAX_RESPONSE = 16_000_000


def request(payload):
    path = os.environ.get("NEURO_BROKER_SOCKET")
    if not path:
        raise RuntimeError("No search feedback broker is configured")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(60)
        connection.connect(path)
        connection.sendall((json.dumps(payload) + "\n").encode())
        with connection.makefile("rb") as stream:
            line = stream.readline(MAX_RESPONSE + 1)
    if len(line) > MAX_RESPONSE or not line.endswith(b"\n"):
        raise RuntimeError("Invalid broker response")
    result = json.loads(line)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Search operation failed"))
    return result["result"]


def format_status(row):
    """Present complete compact feedback without making callers filter JSON."""
    def number(value, *, signed=False):
        if value is None:
            return "unavailable"
        return format(value, "+.6f" if signed else ".6f")

    lines = []
    if row.get("status") == "error":
        lines.append("Failure: " + row.get("error_reason", "Evaluation failed"))
        lines.append("Candidate trace: use --details")
    elif row.get("status") == "ok":
        lines.append(f"Pooled search score: {number(row.get('search_score'))} | "
                     f"parent delta: {number(row.get('parent_score_delta'), signed=True)}")
        for name, entry in row.get("subregion_scores", {}).items():
            lines.append(f"ROI {name}: score {number(entry.get('score'))} | "
                         f"parent delta: {number(entry.get('parent_score_delta'), signed=True)}")
    identity = [f"Model {row.get('id', '?')}", f"status {row.get('status', '?')}"]
    for key, label in (("parent_id", "parent"), ("proposer", "proposer"),
                       ("dataset", "dataset"), ("region", "region"), ("metric", "metric"),
                       ("n_params", "parameters")):
        if row.get(key) is not None:
            identity.append(f"{label} {row[key]}")
    if row.get("seconds") is not None:
        identity.append(f"runtime {row['seconds']:.1f}s")
    lines.append(" | ".join(identity))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("propose-model")
    submit.add_argument("file")
    submit.add_argument("--idea", default="")
    submit.add_argument("--parent", dest="parent_id")
    submit = commands.add_parser("submit")
    submit.add_argument("--file", required=True)
    submit.add_argument("--idea", default="")
    submit.add_argument("--parent", dest="parent_id")
    commands.add_parser("my-best")
    for name in ("random-models", "top-models", "history"):
        command = commands.add_parser(name)
        command.add_argument("-n", "--limit", type=int, default=10)
    for name in ("status", "source", "wait-for-score"):
        command = commands.add_parser(name)
        command.add_argument("id")
        if name in {"status", "wait-for-score"}:
            command.add_argument("--json", action="store_true", help="print structured feedback JSON")
            command.add_argument("--details", action="store_true",
                                 help="include full subject-by-region-by-source-group diagnostics")
    for name in ("peek-island", "summary", "direction"):
        commands.add_parser(name)
    for name in ("set-direction", "reset-island"):
        command = commands.add_parser(name)
        command.add_argument("proposer")
        command.add_argument("--role", "--direction", dest="direction", required=True)
    args = vars(parser.parse_args(argv))
    command = args["command"]
    as_json = args.pop("json", False)
    if command in {"propose-model", "submit"}:
        path = Path(args.pop("file"))
        if path.stat().st_size > 1_000_000:
            parser.error("Source must be at most 1 MB")
        args.update(command="propose-model",
                    source=path.read_text())
        args.pop("reference", None)
        args.pop("spec", None)
    if command == "wait-for-score":
        args["command"] = "status"
        while True:
            result = request(args)
            if result["status"] in {"ok", "error"}:
                break
            time.sleep(5)
    else:
        result = request(args)
    if command in {"status", "wait-for-score"} and not as_json and not args.get("details"):
        print(format_status(result))
    else:
        print(result if isinstance(result, str) else json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
