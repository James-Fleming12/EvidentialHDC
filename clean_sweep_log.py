#!/usr/bin/env python3
"""
clean_sweep_log.py
==================
A log post-processing utility for Section 3 Dual-Uncertainty Gating sweeps.

This script parses raw, verbose log files (such as full_diagnostic_sweep.log)
and strips out all filler (tqdm progress bars, timestamp boilerplate, prototype
rotation dictionaries, per-class purity dictionaries, and initial TP/FP/FN vectors),
preserving ONLY the critical Section 3 headers, Test D0-D7 diagnostic tables,
AUROC scores, and final mIoU/Acc summary lines.

Usage:
    python3 clean_sweep_log.py [input_log_file] [output_summary_file]
    
Example:
    python3 clean_sweep_log.py full_diagnostic_sweep.log clean_sweep_summary.txt
"""

import sys
import os
import re

def clean_log(input_file="full_diagnostic_sweep.log", output_file="clean_sweep_summary.txt"):
    if not os.path.exists(input_file):
        print(f"Error: Input log file '{input_file}' not found.")
        return

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    clean_lines = []
    capture_block = False
    
    # Headers and Section triggers we WANT to preserve
    keep_headers = [
        "Starting Evaluation for Method:",
        "Part 1:",
        "Part 2:",
        "[Test ",
        "Result for ",
        "[Section 3.2]",
        "[Section 3.3]",
        "[Test D0]",
        "[Test D1]",
        "[Test D2]",
        "[Test D4]",
        "[Test D7]",
        "[Section 3.3 / Dynamic Geom]",
        "Saving feature dump",
        "=========================================",
        "-----------------------------------------"
    ]
    
    # Keywords that indicate filler lines or verbose dictionaries we WANT to strip
    ignore_keywords = [
        "tqdm", "it/s", "Adapting:", "Populating Source Stats:",
        "Initial Prototype Norms", "Final Prototype Norms", "Prototype Rotation",
        "Head Rotation:", "Tail Rotation:",
        "Per-Class Veto Purity", "Head Purity:", "Tail Purity:",
        "Per-Class Firing Rates", "Head Firing:", "Tail Firing:",
        "View Disagreement Precision Tracking", "Agreeing Points Precision:",
        "Disagreeing Points Precision:", "Class 3 Precision:", "Class 7 Precision:", "Class 10 Precision:",
        "-> Initial Tail TP:", "-> Initial Tail FP:", "-> Initial Tail FN:",
        "Initializing baseline dataset", "Pre-loading corruption datasets",
        "Resetting model to clean pretrained weights", "Testing snow", "Testing beam_missing", "Testing wet_ground",
        "[DEBUG]"
    ]

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        # Check if line contains any ignore keyword (check before regex stripping)
        if any(ik in line for ik in ignore_keywords):
            capture_block = False
            continue
            
        # Strip timestamp and logger prefix if present (e.g. "2026-07-26 20:19:23,260 - EvalAdapt - INFO -   ")
        # While preserving leading spaces for indented table rows
        cleaned_text = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+-\s+\w+\s+-\s+(INFO|DEBUG|WARNING)\s+-\s*', '', line)
        cleaned_text_strip = cleaned_text.strip()
        
        # Check if line matches a keep header
        if any(kh in cleaned_text for kh in keep_headers) or any(kh in line for kh in keep_headers):
            capture_block = True
            clean_lines.append(cleaned_text.rstrip())
            continue
            
        # If we are in a capture block, keep indented continuation lines (table rows, AUROC values, precision lines)
        if capture_block:
            if cleaned_text.startswith("  ") or cleaned_text.startswith("\t"):
                clean_lines.append(cleaned_text.rstrip())
            else:
                # Non-indented line ends the capture block unless it's a separator
                if "=================" in cleaned_text or "-----------------" in cleaned_text:
                    clean_lines.append(cleaned_text.rstrip())
                else:
                    capture_block = False

    output_content = "\n".join(clean_lines) + "\n"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
        
    print(f"✅ Cleaned summary successfully written to: {output_file}")
    print(f"📊 Preserved {len(clean_lines)} essential diagnostic lines from {len(lines)} raw log lines.")

if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "full_diagnostic_sweep.log"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "clean_sweep_summary.txt"
    clean_log(infile, outfile)
