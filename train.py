"""Final model trainer: Hindi + English specialists + global fallback."""

import csv
import os
import argparse
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from features import load_wav, speech_before, frame_energy_db, f0_contour


def safe_slope(values):
    values = np.asarray(values, dtype=np.float32)

    if len(values) < 3:
        return 0.0

    t = np.arange(len(values), dtype=np.float32)
    return float(np.polyfit(t, values, 1)[0])


def extract_features(x, sr, pause_start, pause_index):

    # CAUSAL: only audio before pause_start
    seg = speech_before(
        x,
        sr,
        pause_start,
        window_s=2.0
    )

    if len(seg) < sr // 10:
        return np.zeros(16, dtype=np.float32)

    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)

    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]

    # ENERGY

    final_energy = (
        float(np.mean(e[-5:]))
        if len(e) >= 5
        else float(np.mean(e))
    )

    mean_energy = float(np.mean(e))
    energy_std = float(np.std(e))

    energy_drop = (
        final_energy - mean_energy
    )

    recent_energy = (
        e[-15:]
        if len(e) >= 15
        else e
    )

    energy_slope = safe_slope(
        recent_energy
    )

    # PITCH

    if len(voiced) >= 3:

        final_pitch = float(
            np.mean(voiced[-3:])
        )

        mean_pitch = float(
            np.mean(voiced)
        )

        pitch_std = float(
            np.std(voiced)
        )

        recent_pitch = (
            voiced[-10:]
            if len(voiced) >= 10
            else voiced
        )

        pitch_slope = safe_slope(
            recent_pitch
        )

        pitch_relative = (
            (final_pitch - mean_pitch)
            /
            (mean_pitch + 1e-6)
        )

    else:

        final_pitch = 0.0
        mean_pitch = 0.0
        pitch_std = 0.0
        pitch_slope = 0.0
        pitch_relative = 0.0

    # VOICING

    voiced_fraction = (
        float(np.mean(voiced_mask))
        if len(f0)
        else 0.0
    )

    recent_mask = (
        voiced_mask[-10:]
        if len(voiced_mask) >= 10
        else voiced_mask
    )

    final_voiced_fraction = (
        float(np.mean(recent_mask))
        if len(recent_mask)
        else 0.0
    )

    final_voiced_run = 0

    for value in voiced_mask[::-1]:

        if value:
            final_voiced_run += 1

        elif final_voiced_run > 0:
            break

    # TURN CONTEXT

    elapsed_time = float(
        pause_start
    )

    pause_position = float(
        pause_index
    )

    context_duration = float(
        len(seg) / sr
    )

    return np.array(
        [
            final_energy,
            mean_energy,
            energy_std,
            energy_drop,
            energy_slope,

            final_pitch,
            mean_pitch,
            pitch_std,
            pitch_slope,
            pitch_relative,

            voiced_fraction,
            final_voiced_fraction,
            float(final_voiced_run),

            elapsed_time,
            pause_position,
            context_duration,
        ],
        dtype=np.float32
    )


def load_dataset(data_dir):

    labels_path = os.path.join(
        data_dir,
        "labels.csv"
    )

    rows = list(
        csv.DictReader(
            open(labels_path)
        )
    )

    cache = {}

    X = []
    y = []

    print(
        f"Loading {data_dir}: "
        f"{len(rows)} pauses"
    )

    for r in rows:

        path = os.path.join(
            data_dir,
            r["audio_file"]
        )

        if path not in cache:
            cache[path] = load_wav(path)

        x, sr = cache[path]

        features = extract_features(
            x,
            sr,
            float(r["pause_start"]),
            int(r["pause_index"])
        )

        X.append(features)

        y.append(
            1 if r["label"] == "eot"
            else 0
        )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(y)
    )


def make_model():

    return make_pipeline(

        StandardScaler(),

        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--english_dir",
        required=True
    )

    parser.add_argument(
        "--hindi_dir",
        required=True
    )

    parser.add_argument(
        "--model_dir",
        default="models"
    )

    args = parser.parse_args()

    os.makedirs(
        args.model_dir,
        exist_ok=True
    )

    # LOAD BOTH LANGUAGES

    X_en, y_en = load_dataset(
        args.english_dir
    )

    X_hi, y_hi = load_dataset(
        args.hindi_dir
    )

    # HINDI SPECIALIST

    print(
        "Training Hindi specialist..."
    )

    hindi_model = make_model()

    hindi_model.fit(
        X_hi,
        y_hi
    )

    joblib.dump(
        hindi_model,
        os.path.join(
            args.model_dir,
            "hindi_model.joblib"
        )
    )

    # ENGLISH SPECIALIST

    print(
        "Training English specialist..."
    )

    english_model = make_model()

    english_model.fit(
        X_en,
        y_en
    )

    joblib.dump(
        english_model,
        os.path.join(
            args.model_dir,
            "english_model.joblib"
        )
    )

    # GLOBAL FALLBACK

    print(
        "Training global fallback..."
    )

    X_global = np.concatenate(
        [
            X_en,
            X_hi
        ],
        axis=0
    )

    y_global = np.concatenate(
        [
            y_en,
            y_hi
        ],
        axis=0
    )

    global_model = make_model()

    global_model.fit(
        X_global,
        y_global
    )

    joblib.dump(
        global_model,
        os.path.join(
            args.model_dir,
            "global_model.joblib"
        )
    )

    print(
        "SUCCESS: saved 3 models to "
        f"{args.model_dir}"
    )

    print(
        "  hindi_model.joblib"
    )

    print(
        "  english_model.joblib"
    )

    print(
        "  global_model.joblib"
    )


if __name__ == "__main__":
    main()