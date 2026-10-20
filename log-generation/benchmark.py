"""
Performance Benchmark Suite for Stage 3 Intent Extraction and Event Predictor.

Measures:
1. Stage-by-stage latency (Extraction, Planning, Enumeration, Graph Construction)
2. End-to-end p50, p95, and p99 latency
3. Cold-start vs. Warm-start latency
4. Peak memory usage via tracemalloc
5. Concurrent throughput (requests/sec) under multi-threading
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import time
import tracemalloc
from typing import Any, Dict, List

try:
    from .engine import Stage3Engine
    from .extractor import MockLLMProvider
    from .schemas import WorkflowStatus
except (ImportError, ValueError):
    from engine import Stage3Engine
    from extractor import MockLLMProvider
    from schemas import WorkflowStatus


def run_benchmarks(num_iterations: int = 100, concurrency: int = 10) -> Dict[str, Any]:
    print(f"=== Starting Stage 3 Performance Benchmark ({num_iterations} runs, concurrency={concurrency}) ===")

    test_prompt = "Create an S3 bucket named analytics-prod-data-lake for company financial reports"
    test_hardened = (
        "TASK: Create S3 bucket named analytics-prod-data-lake. "
        "SECURITY REQUIREMENTS: Require server-side encryption with AES256. Block all public access."
    )

    # 1. Cold Start Benchmark (fresh engine instance)
    tracemalloc.start()
    t0 = time.perf_counter()
    cold_engine = Stage3Engine()
    cold_out = cold_engine.predict(test_prompt, test_hardened, "cold-001")
    t1 = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cold_latency_ms = (t1 - t0) * 1000.0
    cold_memory_kb = peak_mem / 1024.0

    print(f"Cold Start Latency: {cold_latency_ms:.3f} ms | Peak Memory: {cold_memory_kb:.2f} KB | Status: {cold_out.status.value}")

    # 2. Stage-by-stage latency on Warm Engine
    engine = Stage3Engine()
    extraction_times: List[float] = []
    planning_times: List[float] = []
    enumeration_times: List[float] = []
    graph_times: List[float] = []
    total_times: List[float] = []

    for i in range(num_iterations):
        t_start = time.perf_counter()

        # Step A: Intent Extraction
        t_a0 = time.perf_counter()
        intent, constraints, model_ver = engine.extractor.extract(test_prompt, test_hardened)
        t_a1 = time.perf_counter()
        extraction_times.append((t_a1 - t_a0) * 1000.0)

        # Step B: Deterministic Planning
        t_b0 = time.perf_counter()
        plan = engine.planner.plan_workflow(intent, constraints, test_prompt)
        t_b1 = time.perf_counter()
        planning_times.append((t_b1 - t_b0) * 1000.0)

        # Step C: Event Enumeration
        t_c0 = time.perf_counter()
        events = engine.enumerator.enumerate_events(plan.proposed_actions, intent.provided_parameters, intent.parameter_provenance)
        t_c1 = time.perf_counter()
        enumeration_times.append((t_c1 - t_c0) * 1000.0)

        # Step D: Graph Construction
        t_d0 = time.perf_counter()
        graph = engine.graph_builder.build_graph(f"iter-{i}", intent, plan.proposed_actions, events)
        t_d1 = time.perf_counter()
        graph_times.append((t_d1 - t_d0) * 1000.0)

        t_end = time.perf_counter()
        total_times.append((t_end - t_start) * 1000.0)

    # 3. Memory Benchmark during Warm Execution
    tracemalloc.start()
    for _ in range(50):
        engine.predict(test_prompt, test_hardened, "mem-test")
    _, warm_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    warm_memory_kb = warm_peak_mem / 1024.0

    # 4. Concurrency Benchmark
    concurrent_prompts = [
        ("Create an S3 bucket named bucket-" + str(i), "Block all public access. Require server-side encryption.")
        for i in range(num_iterations)
    ]

    t_conc_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(engine.predict, orig, hard, f"conc-{idx}")
            for idx, (orig, hard) in enumerate(concurrent_prompts)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    t_conc_end = time.perf_counter()

    conc_duration_sec = t_conc_end - t_conc_start
    throughput_rps = num_iterations / conc_duration_sec

    # Calculate percentiles
    total_times.sort()
    p50_total = statistics.median(total_times)
    p95_total = total_times[int(0.95 * len(total_times))]
    p99_total = total_times[int(0.99 * len(total_times))]

    metrics = {
        "cold_start": {
            "latency_ms": round(cold_latency_ms, 3),
            "peak_memory_kb": round(cold_memory_kb, 2),
        },
        "stage_latency_mean_ms": {
            "intent_extraction_ms": round(statistics.mean(extraction_times), 4),
            "deterministic_planning_ms": round(statistics.mean(planning_times), 4),
            "event_enumeration_ms": round(statistics.mean(enumeration_times), 4),
            "graph_construction_ms": round(statistics.mean(graph_times), 4),
        },
        "end_to_end_latency_ms": {
            "p50_ms": round(p50_total, 3),
            "p95_ms": round(p95_total, 3),
            "p99_ms": round(p99_total, 3),
            "mean_ms": round(statistics.mean(total_times), 3),
            "min_ms": round(min(total_times), 3),
            "max_ms": round(max(total_times), 3),
        },
        "warm_peak_memory_kb": round(warm_memory_kb, 2),
        "concurrency": {
            "workers": concurrency,
            "total_requests": num_iterations,
            "duration_seconds": round(conc_duration_sec, 4),
            "throughput_rps": round(throughput_rps, 2),
        }
    }

    print("\n=== BENCHMARK RESULTS ===")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run_benchmarks()
