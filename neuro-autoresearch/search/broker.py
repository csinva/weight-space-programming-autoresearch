"""Bounded search API. Session identity belongs to the server, never the request."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socketserver
import tempfile
import threading

import hub

MAX_REQUEST = 1_100_000
MAX_SOURCE = 1_000_000
PUBLIC_FIELDS = ("id", "idea", "experiment_notes", "proposer", "dataset", "status",
                 "test_corr", "roi_corr", "n_params", "model_loc", "seconds", "created_at",
                 "metric", "stage", "protocol_version", "subject_scores", "rsa_diagnostics",
                 "parent_id", "model_sha256", "repair_depth")


def public_row(row, *, details=False):
    if row is None:
        return None
    result = {key: row[key] for key in PUBLIC_FIELDS if key in row}
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
    if isinstance(metadata, dict) and isinstance(metadata.get("candidate_error"), str):
        result["candidate_error"] = metadata["candidate_error"][-1500:]
    if row.get("status") == "error":
        message = str(metadata.get("status", "")) if isinstance(metadata, dict) else ""
        result["error_reason"] = "Evaluation failed; inspect candidate shape, resources and model contract"
        for needles, reason in (
            (("timeout", "timed out"), "Evaluation timed out"),
            (("parameter", "n_params"), "Parameter contract failed"),
            (("nonfinite", "non-finite", "finite"), "Nonfinite model features or score"),
            (("constant",), "Degenerate constant features or score"),
            (("batch", "per-image"), "Per-image independence check failed"),
            (("shape", "width", "dimension"), "Feature shape or width contract failed"),
            (("source identity", "hash"), "Archived model identity check failed"),
            (("cohort", "incompatible evaluation"), "Scientific protocol validation failed"),
            (("isolat", "namespace", "bubblewrap"), "Isolated candidate execution failed"),
        ):
            if any(word in message.lower() for word in needles):
                result["error_reason"] = reason
                break
    for key in ("roi_corr", "subject_scores", "rsa_diagnostics", "experiment_notes"):
        if isinstance(result.get(key), str):
            try:
                result[key] = json.loads(result[key])
            except ValueError:
                pass
    if not details and isinstance(result.get("rsa_diagnostics"), dict):
        diagnostics = result["rsa_diagnostics"]
        result["rsa_diagnostics"] = {key: value for key, value in diagnostics.items()
                                     if key != "subject_results"}
        if isinstance(diagnostics.get("source_groups"), dict):
            groups = {}
            for name, group in diagnostics["source_groups"].items():
                compact = {key: group[key] for key in ("status", "rsa", "n_subjects") if key in group}
                counts = group.get("n_stimuli_by_subject", {})
                if counts and len(set(counts.values())) == 1:
                    compact["n_stimuli"] = next(iter(counts.values()))
                elif counts:
                    compact["n_stimuli_by_subject"] = counts
                compact["regions"] = {
                    region: {key: values[key] for key in ("status", "rsa", "n_subjects") if key in values}
                    for region, values in group.get("regions", {}).items()}
                groups[name] = compact
            result["rsa_diagnostics"]["source_groups"] = groups
    return result


class Broker:
    """One private socket per session; caller must mount only that socket's directory."""

    def __init__(self, run_dir, role, proposer, socket_path, *, reference=None, input_spec=None):
        if role not in {"proposer", "conductor", "simplifier"}:
            raise ValueError("Only proposer, conductor and simplifier sessions have broker access")
        if role in {"proposer", "simplifier"} and not re.fullmatch(r"p[0-9]+", proposer or ""):
            raise ValueError("Invalid proposer identity")
        self.reference_source = self.reference_spec = None
        if role == "simplifier":
            if not reference or not input_spec:
                raise ValueError("Simplifier requires trusted reference and input specification")
            self.reference_source = Path(reference).read_bytes()
            self.reference_spec = Path(input_spec).read_bytes()
            if len(self.reference_source) > MAX_SOURCE or len(self.reference_spec) > 65536:
                raise ValueError("Simplifier reference exceeds bounds")
        self.run = str(Path(run_dir).resolve())
        self.role, self.proposer = role, proposer
        self.socket_path = str(socket_path)
        self.server = None
        self.thread = None

    def _rows(self, sql, args=()):
        with hub.connect(self.run) as con:
            return [dict(row) for row in con.execute(sql, args)]

    def _public(self, row, *, details=False):
        result = public_row(row, details=details)
        if row and row.get("parent_id") and row.get("test_corr") is not None:
            parents = self._rows("SELECT test_corr FROM models WHERE id=? AND metric='rsa' "
                                 "AND stage='search' AND status='ok'", (row["parent_id"],))
            if parents and parents[0]["test_corr"] is not None:
                result["parent_rsa_delta"] = row["test_corr"] - parents[0]["test_corr"]
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
        if not rows or (self.role in {"proposer", "simplifier"} and os.environ.get("VANILLA_AUTORESEARCH") == "1"
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
        conductor_commands = {"top-models", "set-direction", "reset-island"}
        allowed = common | (proposer_commands if self.role in {"proposer", "simplifier"} else conductor_commands)
        if self.role == "simplifier":
            allowed.add("check-equivalence")
        if cmd not in allowed:
            raise PermissionError("Operation is not available to this session")
        limit = request.get("limit", 10)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Limit must be an integer between 1 and 100")
        if cmd == "check-equivalence":
            source = request.get("source")
            if not isinstance(source, str) or not 0 < len(source.encode()) <= MAX_SOURCE:
                raise ValueError("Source must be nonempty and at most 1 MB")
            with tempfile.TemporaryDirectory(prefix="neuro-equivalence-") as directory:
                root = Path(directory)
                (root / "candidate.py").write_text(source)
                (root / "reference.py").write_bytes(self.reference_source)
                (root / "spec.json").write_bytes(self.reference_spec)
                result = hub.check_equivalence(str(root / "candidate.py"),
                                               str(root / "reference.py"), str(root / "spec.json"))
            allowed_fields = {"equivalent", "max_abs_diff", "max_rel_diff", "ref_shape", "cand_shape", "rtol", "atol"}
            feedback = {key: value for key, value in result.items() if key in allowed_fields}
            if "error" in result or "reason" in result:
                feedback["reason"] = "shape mismatch" if result.get("reason") == "shape mismatch" else "Isolated equivalence check failed"
            return feedback
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
                return hub.propose(str(path), idea=idea, proposer=self.proposer, run=self.run,
                                   parent_id=parent_id)
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
                              "AND metric='rsa' AND stage='search' AND created_at>? "
                              "ORDER BY test_corr DESC LIMIT 1", (self.proposer, reset_at))
            if not rows:
                return None
            result = self._public(rows[0])
            if request.get("with_code", True):
                result["code"] = self._source(rows[0])
            return result
        if cmd in {"history", "top-models", "random-models"}:
            if cmd == "random-models" and os.environ.get("VANILLA_AUTORESEARCH") == "1":
                raise PermissionError("Cross-island feedback is disabled")
            where = "metric='rsa' AND stage='search'"
            args = []
            if self.role in {"proposer", "simplifier"} and os.environ.get("VANILLA_AUTORESEARCH") == "1":
                where += " AND proposer=?"
                args.append(self.proposer)
            if cmd != "history":
                where += " AND status='ok'"
            order = {"history": "created_at DESC", "top-models": "test_corr DESC",
                     "random-models": "RANDOM()"}[cmd]
            rows = self._rows(f"SELECT * FROM models WHERE {where} ORDER BY {order} LIMIT ?",
                              (*args, limit))
            result = [self._public(row) for row in rows]
            return result
        if cmd == "peek-island":
            if os.environ.get("VANILLA_AUTORESEARCH") == "1":
                raise PermissionError("Cross-island feedback is disabled")
            islands = self._rows("SELECT DISTINCT proposer FROM models WHERE proposer<>? "
                                 "AND status='ok' AND metric='rsa' AND stage='search' "
                                 "ORDER BY RANDOM() LIMIT 1", (self.proposer,))
            if not islands:
                return None
            rows = self._rows("SELECT idea,experiment_notes FROM models WHERE proposer=? "
                              "AND status='ok' AND metric='rsa' AND stage='search' "
                              "ORDER BY test_corr DESC LIMIT 1", (islands[0]["proposer"],))
            return public_row(rows[0]) if rows else None
        if cmd == "summary":
            where, args = "metric='rsa' AND stage='search'", ()
            if self.role in {"proposer", "simplifier"} and os.environ.get("VANILLA_AUTORESEARCH") == "1":
                where += " AND proposer=?"
                args = (self.proposer,)
            return self._rows(f"SELECT proposer,status,COUNT(*) AS count,MAX(test_corr) AS best_rsa "
                              f"FROM models WHERE {where} GROUP BY proposer,status", args)
        if cmd == "direction":
            if self.role in {"proposer", "simplifier"}:
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
        # hub.reset_island's environment check is not the authorization boundary;
        # this trusted session has already checked its immutable conductor role.
        if os.environ.get("LAB_PROPOSER"):
            raise RuntimeError("Conductor broker must launch without LAB_PROPOSER")
        return hub.reset_island(island.name, role=direction, run=self.run)

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
