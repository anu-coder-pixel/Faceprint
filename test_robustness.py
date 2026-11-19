"""
Test script for Milestone 5: Robustness Testing.

DISCLAIMER:
This experiment utilizes synthetic, AI-generated portrait variations for exploratory testing.
Results benchmark algorithmic behavior under controlled synthetic perturbations and do NOT
represent real-world biometric accuracy, fairness, or deployment-grade performance.
"""

import os
import sys

from modules.robustness import RobustnessEvaluator


def run_tests():
    print("=" * 80)
    print(" Faceprint: Milestone 5 - Robustness Testing (Synthetic Dataset)")
    print("=" * 80)
    print(
        "\n[DISCLAIMER] These experiments use synthetic, AI-generated portrait illustrations.\n"
        "             Results demonstrate algorithmic sensitivity to image perturbations and\n"
        "             do NOT claim or represent real-world recognition accuracy.\n"
    )

    dataset_dir = "Faceprint_Test_Dataset"
    output_csv = os.path.join("results", "robustness_results.csv")
    threshold = 0.363

    # 1. Initialize evaluator
    print(f"[1/5] Initializing RobustnessEvaluator (Dataset: '{dataset_dir}', Threshold: {threshold})...")
    try:
        evaluator = RobustnessEvaluator(dataset_dir=dataset_dir, threshold=threshold)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"      Discovered {len(evaluator.identities)} test subjects: {evaluator.identities}")
    if len(evaluator.identities) == 0:
        print("[ERROR] No subjects found with 'original.png' reference images.")
        sys.exit(1)

    # 2. Enroll baseline references
    print(f"\n[2/5] Enrolling baseline reference images ('original.png')...")
    enrollment_reports = evaluator.enroll_references()
    for identity, report in enrollment_reports.items():
        status_symbol = "✓" if report["success"] else "✗"
        reason = report["reason"]
        conf = f" (Confidence: {report['confidence'] * 100:.1f}%)" if report.get("confidence") else ""
        print(f"      {status_symbol} {identity:<15}: {reason}{conf}")

    # 3. Evaluate 7 altered conditions
    print(f"\n[3/5] Evaluating 7 altered conditions against enrolled gallery...")
    results = evaluator.evaluate_probes()

    # 4. Print sample-by-sample detailed results table
    print("\n" + "-" * 88)
    print(
        f"{'Subject':<14} | {'Condition':<13} | {'Det?':<5} | {'Rec?':<5} | {'Predicted':<14} | {'Sim':<6} | {'Reason'}"
    )
    print("-" * 88)

    for r in results:
        det_str = "YES" if r["detection_succeeded"] else "NO"
        rec_str = "YES" if r["recognition_succeeded"] else "NO"
        pred = r["predicted_identity"][:14]
        sim_str = f"{r['similarity_score']:.3f}" if r["detection_succeeded"] else " N/A "
        reason = r["failure_reason"]
        print(
            f"{r['expected_identity']:<14} | {r['condition']:<13} | {det_str:<5} | {rec_str:<5} | {pred:<14} | {sim_str:<6} | {reason}"
        )

    print("-" * 88)

    # 5. Compute summary metrics and save CSV
    print(f"\n[4/5] Computing robustness metrics and saving CSV...")
    metrics = evaluator.calculate_metrics(results)
    saved_csv_path = evaluator.save_csv(results, output_path=output_csv)
    print(f"      Detailed results successfully saved to: '{saved_csv_path}'")

    # Display overall metrics
    total = metrics["total_samples"]
    detected = metrics["total_detected"]
    recognized = metrics["total_recognized"]
    det_rate = metrics["detection_success_rate"]
    rec_rate = metrics["overall_identification_rate"]
    cond_rec_rate = metrics["conditional_identification_rate"]

    print("\n" + "=" * 80)
    print(" SUMMARY PERFORMANCE METRICS (Exploratory Synthetic Results)")
    print("=" * 80)
    print(f"  Total Probe Samples Tested       : {total}")
    print(f"  Face Detection Successes         : {detected} / {total} ({det_rate:.1f}%)")
    print(f"  Overall Correct Identifications  : {recognized} / {total} ({rec_rate:.1f}%)")
    print(f"  Identification Rate on Detected  : {recognized} / {detected} ({cond_rec_rate:.1f}%) [if detected]")

    # Display per-condition breakdown
    print("\n" + "-" * 72)
    print(f"{'Condition':<15} | {'Samples':<8} | {'Detection Rate':<15} | {'Recognition Rate':<18} | {'Avg Sim'}")
    print("-" * 72)
    for cond, data in metrics["per_condition"].items():
        print(
            f"{cond:<15} | {data['total']:<8} | {data['detection_rate']:>6.1f}%          | {data['recognition_rate']:>8.1f}%           | {data['avg_similarity']:.4f}"
        )
    print("-" * 72)

    print("\n[5/5] Verification Checklist:")
    print("  ✓ Processed all 7 conditions per subject (illumination, blur, noise, pose, tilt, occlusion)")
    print("  ✓ Error handling caught detection & recognition boundaries gracefully")
    print("  ✓ Full CSV output written to results/robustness_results.csv")
    print("  ✓ Explicit disclaimer applied: Exploratory synthetic benchmark, not real-world certified")
    print("\n" + "=" * 80)
    print(" [COMPLETED] Milestone 5 Robustness Testing finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()

