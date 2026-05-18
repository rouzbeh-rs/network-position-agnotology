# Network Position and Manufactured Agnotology

Bayesian agent-based simulations testing whether the structural
position of biased agents within epistemic networks amplifies
their capacity to manufacture ignorance.

## Paper

"Does Network Position Amplify Manufactured Agnotology?"

## Requirements

- Python 3.8+
- numpy, pandas, networkx, scipy, matplotlib, tqdm, pingouin

Install dependencies:
pip install -r requirements.txt

## Usage

Run the full experiment:
python simulation.py

The script runs six experimental phases (core experiment, bias
strength moderation, temporal dynamics, heterogeneous priors,
efficacy differences, and negative controls) totaling ~211,000
simulations. Full runtime is several hours depending on hardware.

## Phases

1. **Core Experiment** — position effect across 8 topologies × 5 sizes
2. **Bias Strength** — moderation across 6 bias levels
3. **Temporal Dynamics** — effect evolution over 50–1,000 rounds
4. **Heterogeneous Priors** — robustness to varied starting credences
5. **Efficacy Differences** — robustness across 5 signal strengths
6. **Negative Controls** — symmetric topologies (Complete, Cycle)

## License

MIT
