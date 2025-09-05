# Explainability Plan and Findings

## Methods
- SHAP (TreeExplainer/KernelExplainer) for global and local attributions.
- LIME for instance-level explanations.

## Outputs
- SHAP summary bar/bee swarm plots
- LIME feature importance for representative samples

## Interpretation Notes
- IEIs with strong positive/negative contributions highlight key indigenous signals.
- Compare 12 vs 24-hour horizons for stability.

## Responsible AI
- Communicate uncertainty; avoid deterministic claims.
- Co-interpret results with domain holders to prevent misrepresentation.
