# End-of-Turn Detection

This project is my submission for the End-of-Turn (EOT) detection task.

The goal is simple: when someone pauses while speaking, decide whether they have actually finished their turn or are just pausing temporarily. The difficult part is that the model has to make this decision causally — it can only use audio that occurred before the pause.

I approached this as an experimentation problem rather than trying to build one complicated model immediately. I started with the provided baseline, added richer prosodic features, tried different classifiers, moved to grouped out-of-fold validation, and finally explored language-specific models for Hindi and English.

## What I tried

The initial model used a few basic signals such as energy before the pause and final pitch. From there, I experimented with a broader set of features describing how the speaker's voice behaves as they approach a pause.

These included:

- Energy level and energy decay near the pause
- Pitch (F0) level and trajectory
- Pitch variability
- Fraction of voiced frames
- Behaviour of the final voiced region
- Time elapsed in the turn
- Position of the pause within the turn

All audio features are computed strictly from audio available before `pause_start`.

I initially used Logistic Regression and later experimented with Histogram Gradient Boosting. I also tried combining the two models and applying additional probability adjustments based on pause position.

Not every experiment helped. In particular, the boosting model and probability ensemble did not improve the main response-delay metric. I have kept these experiments documented in `RUNLOG.md` because they were useful in deciding what the final model should look like.

## Validation

One issue I noticed early was that evaluating a model on the same turns it was trained on gave overly optimistic results.

I therefore switched to grouped out-of-fold validation, where complete turns are kept together. This ensures that the model generating a prediction for a turn has not seen that turn during training.

The best honest OOF results I obtained were:

| Language | AUC | Mean Response Delay | Interrupted Turns |
|---|---:|---:|---:|
| Hindi | 0.654 | 850 ms | 5.0% |
| English | 0.596 | 1301 ms | 5.0% |

The difference between Hindi and English performance led me to experiment with language-specific models instead of forcing both languages through exactly the same trained classifier.

## Final approach

The final system contains three saved Logistic Regression models:

- Hindi specialist
- English specialist
- Global fallback

The models use standardized causal prosodic and turn-context features.

During inference, `predict.py` attempts to identify whether the input is Hindi or English using the available dataset structure/identifiers. It then loads the corresponding specialist model. If the language cannot be determined reliably, it falls back to the global model.

The important point is that `predict.py` only loads already-trained models. It does not train or refit a model on the evaluation data.

## Repository structure

- `predict.py` — inference pipeline
- `train.py` — final model training pipeline
- `features.py` — audio and feature utilities
- `hindi_model.joblib` — trained Hindi specialist
- `english_model.joblib` — trained English specialist
- `global_model.joblib` — trained fallback model
- `RUNLOG.md` — experiments and results
- `NOTES.md` — implementation and design details
- `SUMMARY.html` — final project summary
- `requirements.txt` — Python dependencies

## Running inference

Install the dependencies:

    pip install -r requirements.txt

Then run:

    python predict.py --data_dir <path_to_data> --out predictions.csv

The output contains:

    turn_id,pause_index,p_eot

where `p_eot` is the predicted probability that the candidate pause represents the end of the speaker's turn.

## Training

To retrain the three models:

    python train.py --english_dir <english_data_dir> --hindi_dir <hindi_data_dir> --model_dir models

The training script creates separate Hindi and English models along with a global fallback model.

## Final thoughts

The biggest takeaway from this task was that optimizing EOT detection is not the same as optimizing ordinary classification accuracy.

A model can have reasonable accuracy or AUC and still perform poorly when the actual requirement is to respond as quickly as possible while interrupting fewer than 5% of turns. A lot of the experimentation therefore became about understanding that trade-off rather than simply trying increasingly complex classifiers.

I explored multiple versions of the feature set, Logistic Regression, Gradient Boosting, model ensembling, pause-position adjustments, and different validation strategies. The final solution is deliberately relatively simple, but it is the result of comparing these alternatives rather than choosing the first model that worked.
