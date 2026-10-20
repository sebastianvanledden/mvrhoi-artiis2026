# Look at the all_results.json file
import json
with open("runs/mvrhoi_finetune/sttran_mvrhoi_dw_gaze_cross_yolo/eval/all_results.json") as f:
    results = json.load(f)
    print(f"Number of clips: {len(results)}")
    print(f"Example predictions: {results[0] if results else 'EMPTY'}")

# Look at the first clip's first frame predictions
pred_dist = results[0]['interaction_distribution'][0]
gt_interactions = results[0]['interactions_gt'][0]

print("GT predicates:", [i for i, v in enumerate(gt_interactions) if v > 0])
print("Pred predicates (>0.1):", [i for i, v in enumerate(pred_dist) if v > 0.1])
print("Pred predicates (>0.2):", [i for i, v in enumerate(pred_dist) if v > 0.2])
print("Max pred confidence:", max(pred_dist), "at", pred_dist.index(max(pred_dist)))

# Load predicate names
import json
with open("vidhoi_related/pred_categories.json") as f:
    pred_categories = json.load(f)

print("\n=== Predicate Analysis ===")
print("GT predicates:")
for idx in [7, 20]:
    print(f"  {idx}: {pred_categories[idx]}")

print("\nPredicted predicates (>0.2) with confidence:")
pred_with_conf = [(i, pred_dist[i]) for i in range(len(pred_dist)) if pred_dist[i] > 0.2]
pred_with_conf.sort(key=lambda x: x[1], reverse=True)
for idx, conf in pred_with_conf:
    gt_marker = "✓ GT" if idx in [7, 20] else ""
    print(f"  {idx}: {pred_categories[idx]:20s} conf={conf:.3f} {gt_marker}")

print("\nPredicate 20 (should be predicted):")
print(f"  20: {pred_categories[20]:20s} conf={pred_dist[20]:.3f}")

print("\nCheck more examples:")
for i in range(min(10, len(results))):
    pred_dist = results[i]['interaction_distribution'][0]
    gt_interactions = results[i]['interactions_gt'][0]
    gt_preds = [j for j, v in enumerate(gt_interactions) if v > 0]
    top_preds = sorted(range(len(pred_dist)), key=lambda x: pred_dist[x], reverse=True)[:5]
    print(f"\nExample {i}: GT={gt_preds}, Top5 Pred={top_preds}")