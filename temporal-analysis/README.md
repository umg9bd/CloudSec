# CloudSec Temporal Analysis

This folder contains the temporal anomaly analysis workflow for the BOTSv3-derived CloudTrail dataset. It prepares event sequences using sliding windows and trains an LSTM model for anomaly detection.

## Project Goal

Build a time-aware detection pipeline that:

1. Validates and inspects enriched behavioral security logs.
2. Converts per-user activity into fixed-shape temporal windows.
3. Trains and evaluates an LSTM classifier to predict anomalous behavior.

## Contents

- `botsv3_enriched_features_behavioral.csv`: Main enriched behavioral dataset.
- `capstone-temporal-analyst.ipynb`: Baseline temporal pipeline notebook.
- `capstone-temporal-analyst  v2.ipynb`: Updated notebook with time-based windowing refinements.
- `README.md`: This documentation.

Related data notes are in `../datasets/LINKS.txt`.

## Data Source

The dataset is based on Splunk BOTSv3 synthetic security logs (CloudTrail and network-focused activity).

Reference:
- https://github.com/splunk/botsv3/blob/master

## Workflow Summary

Both notebooks follow the same high-level flow:

1. Load and inspect the dataset.
2. Validate required fields (including temporal and anomaly labels).
3. Clean records (`cloudtrail_time` parsing, null handling, deduplication).
4. Build sliding windows per user.
5. Save training arrays.
6. Train and evaluate an LSTM model in PyTorch.

### Version Differences

- `capstone-temporal-analyst.ipynb`
	- Uses row-count sliding windows with `WINDOW_SIZE = 10`.
	- Creates labels from the next event after each window.

- `capstone-temporal-analyst  v2.ipynb`
	- Uses 10-minute time-based windows.
	- Uses majority-vote window labels.
	- Saves event names per window to JSON.

## Requirements

Use Python 3.10+.

Install dependencies:

```bash
pip install numpy pandas torch jupyter
```

## Running the Analysis

1. Open one of the notebooks in Jupyter/VS Code.
2. Confirm the CSV path in the notebook points to this local file:

```text
temporal-analysis/botsv3_enriched_features_behavioral.csv
```

3. Run cells top-to-bottom.

## Expected Outputs

During notebook execution, the following artifacts are generated in the working directory:

- `X_sequences.npy`: Windowed feature tensor, shape `(N, 10, 19)`.
- `y_labels.npy`: Binary labels for each window.
- `window_summary.txt`: Data cleaning and windowing summary.
- `event_names_per_window.json`: Event names per window (v2 notebook).
- `lstm_model.pt`: Trained PyTorch LSTM model weights.

## Notes

- The notebooks were originally authored in a Kaggle-style environment, so some input paths may need to be updated for local execution.
- For reproducibility, keep window size/duration and feature columns consistent across experiments.