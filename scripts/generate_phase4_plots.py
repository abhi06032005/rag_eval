"""
Generate publication-quality visual plots for Phase 4 deliverables.
Plots:
  1. plots/scifact_benchmark_comparison.png
  2. plots/qdrant_ann_recall_tradeoff.png
  3. plots/concurrency_latency_saturation.png
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    os.makedirs("plots", exist_ok=True)
    plt.style.use("tableau-colorblind10" if "tableau-colorblind10" in plt.style.available else "default")
    
    # 1. SciFact Benchmark Retrieval Metrics Plot
    scifact_csv = "results/scifact/results.csv"
    if os.path.exists(scifact_csv):
        df = pd.read_csv(scifact_csv)
        
        # Abbreviate names for plotting
        labels = [
            "BM25\n(Lexical)",
            "Dense\nExact",
            "Qdrant\nHNSW Std",
            "Qdrant\nHNSW Tuned",
            "Qdrant\nSQ INT8",
            "Qdrant\nBQ 1-bit",
            "Hybrid\nRRF (k=60)",
        ]
        
        x = np.arange(len(labels))
        width = 0.22
        
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
        rects1 = ax.bar(x - width, df["Recall@10"], width, label="Recall@10", color="#1f77b4")
        rects2 = ax.bar(x, df["nDCG@10"], width, label="nDCG@10", color="#2ca02c")
        rects3 = ax.bar(x + width, df["MRR"], width, label="MRR", color="#ff7f0e")
        
        ax.set_ylabel("Evaluation Metric Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
        ax.set_title("BEIR SciFact Official Benchmark: Retrieval Effectiveness Across Paradigms", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylim(0.4, 0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)
        
        # Add values on top of bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
                        
        plt.tight_layout()
        plot1_path = "plots/scifact_benchmark_comparison.png"
        plt.savefig(plot1_path)
        plt.close()
        print(f"Generated {plot1_path}")

        # 2. ANN Recall vs Latency Trade-off
        qdrant_rows = df[df["Configuration"].str.contains("Qdrant|Exact")].copy()
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        
        ax.scatter(qdrant_rows["Latency_mean_ms"], qdrant_rows["ANN_Recall@10"], s=120, c="#d62728", zorder=3)
        for _, row in qdrant_rows.iterrows():
            lbl = row["Configuration"].replace("Qdrant ", "").replace(" (Standalone Rust Server)", "")
            ax.annotate(lbl, xy=(row["Latency_mean_ms"], row["ANN_Recall@10"]), xytext=(5, -5),
                        textcoords="offset points", fontsize=8.5, fontweight="medium")
                        
        ax.set_xlabel("Mean Search Latency (ms)", fontsize=11, fontweight="bold")
        ax.set_ylabel("ANN Recall@10 (vs. Exact Ground Truth)", fontsize=11, fontweight="bold")
        ax.set_title("Qdrant Standalone Engine: ANN Recall vs. Retrieval Latency Trade-off", fontsize=12, fontweight="bold")
        ax.set_ylim(0.975, 1.002)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot2_path = "plots/qdrant_ann_recall_tradeoff.png"
        plt.savefig(plot2_path)
        plt.close()
        print(f"Generated {plot2_path}")

    # 3. Concurrency Saturation Plot
    load_json = "results/load_test_results.json"
    if os.path.exists(load_json):
        with open(load_json, "r", encoding="utf-8") as f:
            load_data = json.load(f)
            
        entries = list(load_data.values())
        concurrency = [entry["concurrency"] for entry in entries]
        rps = [entry["throughput_rps"] for entry in entries]
        p95 = [entry["p95_ms"] for entry in entries]
        p50 = [entry["p50_ms"] for entry in entries]
        
        fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
        
        color = "#1f77b4"
        ax1.set_xlabel("Concurrent Client Workers", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Throughput (Requests / Second)", color=color, fontsize=11, fontweight="bold")
        line1 = ax1.plot(concurrency, rps, marker="o", linewidth=2.5, color=color, label="Throughput (RPS)")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.4)
        
        ax2 = ax1.twinx()
        color = "#d62728"
        ax2.set_ylabel("Latency (ms)", color=color, fontsize=11, fontweight="bold")
        line2 = ax2.plot(concurrency, p95, marker="s", linestyle="--", linewidth=2.0, color="#d62728", label="p95 Latency")
        line3 = ax2.plot(concurrency, p50, marker="^", linestyle=":", linewidth=2.0, color="#ff7f0e", label="p50 Latency")
        ax2.tick_params(axis="y", labelcolor=color)
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right", frameon=True)
        
        plt.title("System Saturation Curve: Throughput vs Tail Latency under Concurrency Sweep", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plot3_path = "plots/concurrency_latency_saturation.png"
        plt.savefig(plot3_path)
        plt.close()
        print(f"Generated {plot3_path}")

if __name__ == "__main__":
    main()
