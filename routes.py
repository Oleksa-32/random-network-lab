from __future__ import annotations

import io
import os
import json
from statistics import mean, pstdev
from typing import Dict, Any, List

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import networkx as nx

from models import db, Study
from config import ALLOWED_EDGE_EXT, POPULAR_DIR, POPULAR_NETWORKS
from network_parsing import parse_txt_network
from metrics import (
    compute_metrics_single,
    centrality_vectors,
    corr_matrix_nodewise,
    flatten_metric_for_group,
    corr_matrix_groupwise,
    graph_to_d3,
)
from mongo_storage import save_graph_to_mongo, load_graph_from_mongo, load_graph_d3_from_mongo


def register_routes(app: Flask, graphs_coll):

    @app.route("/")
    def index():
        popular_slug = request.args.get("popular")
        selected_popular = None
        if popular_slug:
            selected_popular = next(
                (x for x in POPULAR_NETWORKS if x["slug"] == popular_slug),
                None,
            )
        return render_template("index.html", popular=selected_popular)

    @app.get("/popular")
    def popular_page():
        items = []
        for item in POPULAR_NETWORKS:
            path = os.path.join(POPULAR_DIR, item["filename"])
            if os.path.isfile(path):
                items.append(item)
        return render_template("popular.html", items=items)

    @app.post("/analyze")
    def analyze():
        mode = request.form.get("mode", "single")
        generator = request.form.get("generator")
        metrics = request.form.getlist("metrics")

        want_corr_nodewise = request.form.get("corr_nodewise") == "on"
        want_corr_group = request.form.get("corr_group") == "on"

        if generator == "popular_file":
            popular_slug = request.form.get("popular_slug")
            item = next((x for x in POPULAR_NETWORKS if x["slug"] == popular_slug), None)
            if not item:
                flash("Unknown popular network.", "error")
                return redirect(url_for("index"))

            path = os.path.join(POPULAR_DIR, item["filename"])
            if not os.path.isfile(path):
                flash("File for this popular network not found.", "error")
                return redirect(url_for("index"))

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            try:
                G = parse_txt_network(content)
            except ValueError as e:
                flash(f"Could not parse popular network file: {e}", "error")
                return redirect(url_for("index"))

            result = compute_metrics_single(G, metrics)

            if want_corr_nodewise:
                cv = centrality_vectors(G)
                corr = corr_matrix_nodewise(cv)
                result.setdefault("_correlations", {})["nodewise_centralities"] = corr

            params = {
                "source": "popular",
                "slug": popular_slug,
                "filename": item["filename"],
                "title": item["title"],
            }

            study = Study(
                mode="single",
                generator="popular",
                params=params,
                metrics=metrics,
                results=result,
            )
            db.session.add(study)
            db.session.commit()

            save_graph_to_mongo(graphs_coll, study.id, G)

            graph_json = json.dumps(graph_to_d3(G))
            return render_template("results.html", study=study, graph_json=graph_json)

        if generator == "file":
            if mode != "single":
                flash("File upload is only supported in single mode.", "error")
                return redirect(url_for("index"))

            file = request.files.get("file")
            if not file or file.filename == "":
                flash("No file provided.", "error")
                return redirect(url_for("index"))

            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EDGE_EXT:
                flash("Supported file types: .txt, .csv, .edgelist", "error")
                return redirect(url_for("index"))

            content_bytes = file.read()
            content = content_bytes.decode("utf-8", errors="ignore")
            try:
                G = parse_txt_network(content)
            except ValueError as e:
                flash(f"Could not parse txt network: {e}", "error")
                return redirect(url_for("index"))

            result = compute_metrics_single(G, metrics)

            if want_corr_nodewise:
                cv = centrality_vectors(G)
                corr = corr_matrix_nodewise(cv)
                result.setdefault("_correlations", {})["nodewise_centralities"] = corr

            params = {
                "source": "file",
                "filename": filename,
            }

            study = Study(
                mode="single",
                generator="file",
                params=params,
                metrics=metrics,
                results=result,
            )
            db.session.add(study)
            db.session.commit()

            save_graph_to_mongo(graphs_coll, study.id, G)

            graph_json = json.dumps(graph_to_d3(G))
            return render_template("results.html", study=study, graph_json=graph_json)

        try:
            if generator == "er":
                n = int(request.form.get("er_n", "100"))
                p = float(request.form.get("er_p", "0.05"))
                params = {"n": n, "p": p}
                gen_fn = lambda: nx.erdos_renyi_graph(n, p, seed=None, directed=False)
            elif generator == "ws":
                n = int(request.form.get("ws_n", "100"))
                k = int(request.form.get("ws_k", "4"))
                p = float(request.form.get("ws_p", "0.1"))
                params = {"n": n, "k": k, "p": p}
                gen_fn = lambda: nx.watts_strogatz_graph(n, k, p, seed=None)
            elif generator == "ba":
                n = int(request.form.get("ba_n", "100"))
                m = int(request.form.get("ba_m", "2"))
                params = {"n": n, "m": m}
                gen_fn = lambda: nx.barabasi_albert_graph(n, m, seed=None)
            else:
                flash("Unknown generator type.", "error")
                return redirect(url_for("index"))
        except ValueError:
            flash("Check parameters: must be numeric.", "error")
            return redirect(url_for("index"))

        if mode == "single":
            G = gen_fn()
            result = compute_metrics_single(G, metrics)

            if want_corr_nodewise:
                cv = centrality_vectors(G)
                corr = corr_matrix_nodewise(cv)
                result.setdefault("_correlations", {})["nodewise_centralities"] = corr

            study = Study(
                mode="single",
                generator=generator,
                params=params,
                metrics=metrics,
                results=result,
            )
            db.session.add(study)
            db.session.commit()

            save_graph_to_mongo(graphs_coll, study.id, G)

            graph_json = json.dumps(graph_to_d3(G))
            return render_template("results.html", study=study, graph_json=graph_json)

        elif mode == "group":
            count = int(request.form.get("group_count", "10"))
            agg_buckets: Dict[str, List[float | None]] = {m: [] for m in metrics}

            preview_G = gen_fn()
            preview_graph_json = None
            if preview_G.number_of_nodes() <= 800:
                preview_graph_json = json.dumps(graph_to_d3(preview_G))

            for _ in range(count):
                G = gen_fn()
                single = compute_metrics_single(G, metrics)
                for m in metrics:
                    val = flatten_metric_for_group(single[m])
                    agg_buckets[m].append(val)

            result: Dict[str, Any] = {}
            for m, arr in agg_buckets.items():
                vals = [float(x) for x in arr if x is not None]
                if len(vals) == 0:
                    result[m] = {"mean": None, "std": None, "samples": 0}
                elif len(vals) == 1:
                    result[m] = {"mean": float(vals[0]), "std": 0.0, "samples": 1}
                else:
                    result[m] = {
                        "mean": float(mean(vals)),
                        "std": float(pstdev(vals)),
                        "samples": len(vals),
                    }

            if want_corr_group and len(metrics) >= 2:
                corr = corr_matrix_groupwise(agg_buckets)
                if corr:
                    result.setdefault("_correlations", {})["group_metrics"] = corr

            params["count"] = count
            study = Study(
                mode="group",
                generator=generator,
                params=params,
                metrics=metrics,
                results=result,
            )
            db.session.add(study)
            db.session.commit()

            save_graph_to_mongo(graphs_coll, study.id, preview_G)

            return render_template("results.html", study=study, graph_json=preview_graph_json)

        else:
            flash("Unknown mode.", "error")
            return redirect(url_for("index"))

    @app.get("/studies")
    def studies():
        items = Study.query.order_by(Study.created_at.desc()).all()
        return render_template("studies.html", items=items)

    @app.get("/studies/<int:study_id>")
    def study_detail(study_id: int):
        study = Study.query.get_or_404(study_id)
        d3_data = load_graph_d3_from_mongo(graphs_coll, study_id)
        graph_json = json.dumps(d3_data) if d3_data is not None else None
        return render_template("results.html", study=study, graph_json=graph_json)

    @app.get("/api/studies/<int:study_id>.json")
    def study_export(study_id: int):
        study = Study.query.get_or_404(study_id)
        payload = {
            "id": study.id,
            "created_at": study.created_at.isoformat(),
            "mode": study.mode,
            "generator": study.generator,
            "params": study.params,
            "metrics": study.metrics,
            "results": study.results,
        }
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return send_file(
            io.BytesIO(data),
            mimetype="application/json; charset=utf-8",
            as_attachment=True,
            download_name=f"study_{study.id}.json",
        )

    @app.get("/api/studies/<int:study_id>/network.txt")
    def study_network_file(study_id: int):
        _ = Study.query.get_or_404(study_id)
        G = load_graph_from_mongo(graphs_coll, study_id)
        if G is None:
            return {"error": "graph_not_found"}, 404

        buf = io.StringIO()
        for u, v in G.edges():
            buf.write(f"{u} {v}\n")
        data = buf.getvalue().encode("utf-8")

        return send_file(
            io.BytesIO(data),
            mimetype="text/plain; charset=utf-8",
            as_attachment=True,
            download_name=f"network_{study_id}.edgelist.txt",
        )

    @app.get("/api/studies/<int:study_id>/graph.json")
    def study_graph(study_id: int):
        d3_data = load_graph_d3_from_mongo(graphs_coll, study_id)
        if d3_data is None:
            return {"error": "graph_not_found"}, 404
        return app.response_class(
            response=json.dumps(d3_data),
            status=200,
            mimetype="application/json",
        )
