"""Final inference script for End-of-Turn detection."""

import argparse
import csv
import os
import joblib
import numpy as np

from features import load_wav, speech_before, frame_energy_db, f0_contour


def safe_slope(values):
    values = np.asarray(values, dtype=np.float32)

    if len(values) < 3:
        return 0.0

    t = np.arange(len(values), dtype=np.float32)
    return float(np.polyfit(t, values, 1)[0])


def extract_features(x, sr, pause_start, pause_index):

    # STRICTLY CAUSAL:
    # only audio before pause_start is used.
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

    energy_drop = final_energy - mean_energy

    recent_energy = (
        e[-15:]
        if len(e) >= 15
        else e
    )

    energy_slope = safe_slope(recent_energy)

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

    elapsed_time = float(pause_start)
    pause_position = float(pause_index)
    context_duration = float(len(seg) / sr)

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


def detect_language(data_dir, rows):

    # First use folder name if available.
    folder = os.path.basename(
        os.path.normpath(data_dir)
    ).lower()

    if "hindi" in folder:
        return "hindi"

    if "english" in folder:
        return "english"

    # Then inspect turn/audio identifiers.
    for row in rows[:20]:

        tid = row.get(
            "turn_id",
            ""
        ).lower()

        audio = row.get(
            "audio_file",
            ""
        ).lower()

        if tid.startswith("hi") or audio.startswith("hi"):
            return "hindi"

        if tid.startswith("en") or audio.startswith("en"):
            return "english"

    # Unknown language -> global model.
    return "global"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True
    )

    parser.add_argument(
        "--model_dir",
        default="models"
    )

    parser.add_argument(
        "--out",
        default="predictions.csv"
    )

    args = parser.parse_args()

    labels_path = os.path.join(
        args.data_dir,
        "labels.csv"
    )

    rows = list(
        csv.DictReader(
            open(labels_path)
        )
    )

    language = detect_language(
        args.data_dir,
        rows
    )

    print(
        f"Detected language: {language}"
    )

    model_filename = {
        "hindi": "hindi_model.joblib",
        "english": "english_model.joblib",
        "global": "global_model.joblib",
    }[language]

    model_path = os.path.join(
        args.model_dir,
        model_filename
    )

    model = joblib.load(
        model_path
    )

    print(
        f"Loaded model: {model_path}"
    )

    cache = {}

    X = []
    keys = []

    for r in rows:

        path = os.path.join(
            args.data_dir,
            r["audio_file"]
        )

        if path not in cache:
            cache[path] = load_wav(path)

        x, sr = cache[path]

        feature_vector = extract_features(
            x,
            sr,
            float(r["pause_start"]),
            int(r["pause_index"])
        )

        X.append(
            feature_vector
        )

        keys.append(
            (
                r["turn_id"],
                r["pause_index"]
            )
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    with open(
        args.out,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "turn_id",
                "pause_index",
                "p_eot"
            ]
        )

        for (
            tid,
            pause_idx
        ), probability in zip(
            keys,
            probabilities
        ):

            writer.writerow(
                [
                    tid,
                    pause_idx,
                    f"{probability:.6f}"
                ]
            )

    print(
        f"wrote {len(keys)} predictions "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()