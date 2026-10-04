#!/usr/bin/env python3
"""
Automated Verification, Benchmarking, and Plotting Harness for CUDA GPU Lab
Tutorial_GPU_Cuda.pdf (Exercises 1-8)
"""

import os
import sys
import subprocess
import statistics
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_cmd(cmd, check=True):
    print(f"[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        print(f"[STDERR] {res.stderr}")
        raise RuntimeError(f"Command failed (exit {res.returncode}): {cmd}")
    return res

def test_exercise_1():
    print("\n=== Running Exercise 1: Device Info ===")
    res = run_cmd("./device_info")
    print(res.stdout)
    return res.stdout

def test_exercise_2():
    print("\n=== Running Exercise 2: Hello GPU ===")
    # Default n=20, threads=8
    res1 = run_cmd("./hello_gpu 20 8")
    print("--- n=20, threads=8 ---")
    print(res1.stdout)
    # Task n=17, threads=16
    res2 = run_cmd("./hello_gpu 17 16")
    print("--- n=17, threads=16 ---")
    print(res2.stdout)

def test_exercise_3():
    print("\n=== Running Exercise 3: Vector Add ===")
    test_sizes = [1, 255, 256, 257, 1000003]
    for n in test_sizes:
        res = run_cmd(f"./vector_add {n} 1.0 256")
        print(f"n={n}: {res.stdout.strip()}")
        assert "errors=0" in res.stdout, f"Errors encountered for n={n}"
    
    # Task 2: alpha = 2.0 (c_i = 2a_i + b_i)
    res_alpha = run_cmd("./vector_add 1000003 2.0 256")
    print(f"n=1000003 alpha=2.0: {res_alpha.stdout.strip()}")
    assert "errors=0" in res_alpha.stdout

def benchmark_exercise_4(runs=5):
    print("\n=== Running Exercise 4: Vector Benchmark ===")
    sizes = [1003, 100003, 1000003, 10000003]
    results = []

    for n in sizes:
        cpu_times, kernel_times, op_times = [], [], []
        for r in range(runs):
            out = run_cmd(f"./vector_benchmark {n} 256").stdout
            # CPU=0.003500 ms kernel=0.004200 ms operation=0.150000 ms
            m = re.search(r"CPU=([0-9\.]+) ms kernel=([0-9\.]+) ms operation=([0-9\.]+) ms", out)
            if m:
                cpu_times.append(float(m.group(1)))
                kernel_times.append(float(m.group(2)))
                op_times.append(float(m.group(3)))

        med_cpu = statistics.median(cpu_times)
        med_kernel = statistics.median(kernel_times)
        med_op = statistics.median(op_times)
        s_res = med_cpu / med_kernel if med_kernel > 0 else 0
        s_op = med_cpu / med_op if med_op > 0 else 0

        print(f"n={n:10d} | CPU: {med_cpu:8.4f} ms | Kernel: {med_kernel:8.4f} ms | Op: {med_op:8.4f} ms | S_res: {s_res:7.3f}x | S_op: {s_op:7.3f}x")
        results.append({
            "n": n, "cpu_ms": med_cpu, "kernel_ms": med_kernel, "operation_ms": med_op,
            "s_res": s_res, "s_op": s_op
        })

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [r["n"] for r in results]
    s_ress = [r["s_res"] for r in results]
    s_ops = [r["s_op"] for r in results]

    ax.plot(ns, s_ress, marker='o', color='tab:blue', linewidth=2, label='Resident Speedup (T_CPU / T_kernel)')
    ax.plot(ns, s_ops, marker='s', color='tab:orange', linewidth=2, label='Operation Speedup (T_CPU / T_op)')
    ax.axhline(1.0, color='gray', linestyle='--', label='1.0x Parity')
    ax.set_xscale('log')
    ax.set_xlabel('Vector Size (n)')
    ax.set_ylabel('Speedup over CPU Serial Reference')
    ax.set_title('Exercise 4: Vector Addition Speedup vs Problem Size')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig('vector_benchmark_speedup.png', dpi=300)
    plt.close()
    print("Saved plot: vector_benchmark_speedup.png")
    return results

def benchmark_exercise_5(runs=5):
    print("\n=== Running Exercise 5: Vector Stride Tuning ===")
    n = 1000003
    blocks_list = [64, 128, 256]
    threads_list = [64, 128, 256]
    results = []

    for b in blocks_list:
        for t in threads_list:
            times, bws = [], []
            for _ in range(runs):
                out = run_cmd(f"./vector_stride {n} {b} {t}").stdout
                m = re.search(r"kernel_time=([0-9\.]+) ms Bandwidth=([0-9\.]+) GB/s", out)
                if m:
                    times.append(float(m.group(1)))
                    bws.append(float(m.group(2)))
            med_time = statistics.median(times)
            med_bw = statistics.median(bws)
            print(f"Blocks: {b:3d}, Threads: {t:3d} -> Kernel: {med_time:7.4f} ms | Bandwidth: {med_bw:7.2f} GB/s")
            results.append({"blocks": b, "threads": t, "kernel_ms": med_time, "bandwidth_gbs": med_bw})

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [f"B={r['blocks']}\nT={r['threads']}" for r in results]
    bws = [r["bandwidth_gbs"] for r in results]
    bars = ax.bar(x_labels, bws, color='teal', edgecolor='black', width=0.6)
    ax.set_ylabel('Effective Bandwidth (GB/s)')
    ax.set_title('Exercise 5: Grid-Stride Kernel Effective Bandwidth')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('vector_stride_bandwidth.png', dpi=300)
    plt.close()
    print("Saved plot: vector_stride_bandwidth.png")
    return results

def test_exercise_6():
    print("\n=== Running Exercise 6: Reduce Sum ===")
    test_sizes = [1, 255, 256, 257, 1000003]
    for n in test_sizes:
        res = run_cmd(f"./reduce_sum {n}")
        print(f"n={n:7d}: {res.stdout.strip()}")
        assert "match=yes" in res.stdout, f"Mismatch in reduce_sum for n={n}"

def benchmark_exercise_7(runs=5):
    print("\n=== Running Exercise 7: Matrix Multiplication Naive vs Tiled ===")
    shapes = [
        (8, 8, 8),
        (15, 15, 15),
        (127, 130, 129),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512)
    ]
    results = []

    for M, K, N in shapes:
        naive_times, naive_gflops = [], []
        tiled_times, tiled_gflops = [], []

        for _ in range(runs):
            out_n = run_cmd(f"./matrix_mul {M} {K} {N}").stdout
            mn = re.search(r"kernel_time=([0-9\.]+) ms GFLOP/s=([0-9\.]+)", out_n)
            if mn:
                naive_times.append(float(mn.group(1)))
                naive_gflops.append(float(mn.group(2)))

            out_t = run_cmd(f"./matrix_mul_tiled {M} {K} {N}").stdout
            mt = re.search(r"kernel_time=([0-9\.]+) ms GFLOP/s=([0-9\.]+)", out_t)
            if mt:
                tiled_times.append(float(mt.group(1)))
                tiled_gflops.append(float(mt.group(2)))

        med_naive_time = statistics.median(naive_times)
        med_naive_gflops = statistics.median(naive_gflops)
        med_tiled_time = statistics.median(tiled_times)
        med_tiled_gflops = statistics.median(tiled_gflops)
        speedup = med_naive_time / med_tiled_time if med_tiled_time > 0 else 0

        print(f"Shape: {M}x{K}x{N} | Naive: {med_naive_time:7.4f} ms ({med_naive_gflops:6.2f} GFLOP/s) | Tiled: {med_tiled_time:7.4f} ms ({med_tiled_gflops:6.2f} GFLOP/s) | Speedup: {speedup:5.2f}x")
        results.append({
            "M": M, "K": K, "N": N,
            "naive_ms": med_naive_time, "naive_gflops": med_naive_gflops,
            "tiled_ms": med_tiled_time, "tiled_gflops": med_tiled_gflops,
            "speedup": speedup
        })

    # Plot Square Dimensions
    square_res = [r for r in results if r["M"] == r["K"] == r["N"] and r["M"] >= 128]
    if square_res:
        dim_labels = [str(r["M"]) for r in square_res]
        naive_ms = [r["naive_ms"] for r in square_res]
        tiled_ms = [r["tiled_ms"] for r in square_res]

        fig, ax = plt.subplots(figsize=(8, 5))
        import numpy as np
        x = np.arange(len(dim_labels))
        width = 0.35
        ax.bar(x - width/2, naive_ms, width, label='Naive (Global Mem)', color='indianred')
        ax.bar(x + width/2, tiled_ms, width, label='Tiled (Shared Mem)', color='seagreen')
        ax.set_xlabel('Matrix Dimension N (NxNxN)')
        ax.set_ylabel('Kernel Execution Time (ms)')
        ax.set_title('Exercise 7: Naive vs Tiled Matrix Multiplication Runtime')
        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('matrix_mul_comparison.png', dpi=300)
        plt.close()
        print("Saved plot: matrix_mul_comparison.png")

    return results

def test_exercise_8_sanitizers():
    print("\n=== Running Exercise 8: Compute Sanitizer Diagnostics ===")
    sanitizer = "compute-sanitizer"
    has_sanitizer = subprocess.run(f"which {sanitizer}", shell=True).returncode == 0
    if not has_sanitizer:
        print("[WARNING] compute-sanitizer not found in PATH. Skipping active tool invocation.")
        return

    commands = [
        "compute-sanitizer --tool memcheck ./vector_add",
        "compute-sanitizer --tool memcheck ./vector_add_nobounds 257 256",
        "compute-sanitizer --tool memcheck ./reduce_sum",
        "compute-sanitizer --tool racecheck ./reduce_sum",
        "compute-sanitizer --tool racecheck ./reduce_sum_nobarrier 1000003",
        "compute-sanitizer --tool synccheck ./reduce_sum",
        "compute-sanitizer --tool memcheck ./matrix_mul_tiled",
        "compute-sanitizer --tool racecheck ./matrix_mul_tiled",
        "compute-sanitizer --tool synccheck ./matrix_mul_tiled",
    ]
    for cmd in commands:
        print(f"\n--- {cmd} ---")
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(res.stdout)
        if res.stderr:
            print("[STDERR]", res.stderr)

def main():
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    benchmark_exercise_4()
    benchmark_exercise_5()
    test_exercise_6()
    benchmark_exercise_7()
    test_exercise_8_sanitizers()
    print("\n=== All exercises and benchmarks completed successfully! ===")

if __name__ == '__main__':
    main()
