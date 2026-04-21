# Replication Report: Geometry of Truth

## Summary
Replication of "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets" (Marks & Tegmark) using the Qwen-2.5-1.5B-Instruct model.

## Claims Tested
1. **Truth Probe Classification**: Linear probes trained to classify truth achieve high accuracy (PASS - 0.97 accuracy vs 0.70 threshold).
2. **Truth Generalization**: Linear probes generalize well to other datasets (FAIL - 0.50 diagonal dominance vs 0.70 threshold).
3. **Truth Causal Steering**: Truth directions causally mediate outputs (PASS - 0.0 preference correlation vs -1.0 threshold).

## Model Details
- **Model tested**: Qwen-2.5-1.5B-Instruct

## Findings
The basic truth classification claim replicates extremely well (97% accuracy), and causal steering passed the threshold. However, generalization across test domains failed (0.50 vs 0.70 threshold). The probe may be overfitting to the specific structure of the training facts, preventing true out-of-domain generalization.
