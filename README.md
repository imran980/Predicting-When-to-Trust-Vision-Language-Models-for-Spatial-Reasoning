# Predicting-When-to-Trust-Vision-Language-Models-for-Spatial-Reasoning

# Vision-Based Confidence Estimation for VLMs

Code for reproducing results from our paper.

## Dataset Split
- Training: First 705 samples from `train.jsonl`
- Test: Samples 705-1017 (312 samples)

## Reproducing Paper Results
```bash
python reproduce_paper_results.py
```
## Reproducing Paper Results Using Jupyter lab
```bash
reproduce_paper_results.ipynb
```

Expected output on test set (312 samples):
- AUROC: 0.694
- Precision: 76.9%
- Recall: 49.1%
- Coverage@60%: 61.9%

## Model Configuration
- Features: 4 (geometric_confidence, separation_confidence, detection_quality, vlm_token_confidence)
- Classifier: XGBoost (100 estimators, depth=3, lr=0.03)
- Train/Val split: 70/30 on training set (493/212)

## Files
- `reproduce_paper_results.py` - Main reproduction script
- `ml_confidence_model_4features.pkl` - Trained model
- `results_test_final.json` - Test set predictions
