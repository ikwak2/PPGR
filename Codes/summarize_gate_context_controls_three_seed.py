#!/usr/bin/env python3
"""Validate and summarize matched C10 context controls across three seeds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from train import state_digest

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/FINAL_260909_GATE_CONTEXT_CONTROL_THREE_SEED"
ROOTS = {
    0: HERE / "results/FINAL_260802_FULL_240_540",
    5000: HERE / "results/FINAL_260802_FULL_240_540_SEED1_OFFSET5000",
    10000: HERE / "results/FINAL_260802_FULL_240_540_SEED1",
}
SOURCES = {"PPG+ACC": "ppg_acc", "PPG+ACC+Temp MLP": "ppg_acc_temp_mlp"}
MODES = ("real", "context_free", "context_shuffled")
HORIZONS = (0, 1, 2, 5, 10)
DURATIONS = (10, 20, 30)
TOP_M = (1, 3, 5, 10)
torch.set_num_threads(1)


def load_json(path):
    return json.loads(path.read_text())


def load_checkpoint(path):
    cp = torch.load(path, map_location="cpu", weights_only=False)
    assert state_digest(cp["model_state"]) == cp["model_state_sha256"], path
    return cp


def macro(payload, horizon):
    cells = payload["calibration_epochs"][str(horizon)]["cells"]
    assert set(cells) == {str(s) for s in DURATIONS}
    assert all(set(cells[str(s)]) == {str(m) for m in TOP_M} for s in DURATIONS)
    value = statistics.mean(cells[str(s)][str(m)]["overall"]["eer"] for s in DURATIONS for m in TOP_M)
    stored = payload["calibration_epochs"][str(horizon)]["oob_macro_eer_across_16_cells"]
    assert abs(value-stored) < 1e-12
    return value * 100


def summarize(values):
    assert len(values) == 3
    return {
        "offset_0_percent": values[0], "offset_5000_percent": values[1],
        "offset_10000_percent": values[2], "mean_percent": statistics.mean(values),
        "seed_sample_sd_percentage_points": statistics.stdev(values),
    }


def write_csv(filename, rows):
    with (OUTPUT/filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--offsets", type=int, nargs="+", choices=list(ROOTS), default=list(ROOTS))
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    if not args.validate_only and args.offsets != list(ROOTS):
        parser.error("Three-seed reporting requires offsets 0 5000 10000 in that order")
    if not args.validate_only and args.sources != list(SOURCES):
        parser.error("Final reporting requires both prespecified sources")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = {}
    audits = []
    execution_audits = []
    fold_rows = []
    for display, source in SOURCES.items():
        if display not in args.sources:
            continue
        for offset, root in ROOTS.items():
            if offset not in args.offsets:
                continue
            real_dir = root / f"{source}_residual_verification_gate"
            for fold in range(1, 5):
                real = load_json(real_dir/f"oob_fold{fold}.json")
                reference_c00 = load_checkpoint(real_dir/"checkpoints"/f"fold{fold}_C00.pt")
                source_path = root/source/"checkpoints"/f"fold{fold}_E50.pt"
                source_cp = load_checkpoint(source_path)
                assert source_cp["model_state_sha256"] == reference_c00["source_model_state_sha256"]
                expected_train = real["train_users"]
                expected_oob = real["oob_users"]
                assert not set(expected_train) & set(expected_oob)
                for mode in MODES:
                    directory = real_dir if mode == "real" else real_dir.with_name(real_dir.name+"_"+mode)
                    payload = load_json(directory/f"oob_fold{fold}.json")
                    assert payload["status"] == "complete", directory
                    assert payload["primary_calibration_epoch"] == 10
                    assert payload["train_users"] == expected_train and payload["oob_users"] == expected_oob
                    assert {key:value for key,value in payload["protocol"].items() if key != "verification_gate_calibration"} == {key:value for key,value in real["protocol"].items() if key != "verification_gate_calibration"}, (directory,"different base protocol")
                    assert payload["protocol"]["fold_seeds"] == real["protocol"]["fold_seeds"]
                    assert payload["protocol"]["normalization"] == real["protocol"]["normalization"]
                    settings = payload["protocol"]["verification_gate_calibration"]
                    real_settings = real["protocol"]["verification_gate_calibration"]
                    for key in ["source_epoch", "calibration_epochs", "primary_horizon", "enrollment_seconds_used_for_calibration", "top_m_values_used_in_loss", "learning_rate", "identity_regularization", "impostor_owners_per_target_activity_step"]:
                        assert settings[key] == real_settings[key], (directory,key)
                    assert settings["oob_used_for_training_or_selection"] is False
                    if mode != "real":
                        metadata = load_json(directory/f"fold{fold}_context_control.json")
                        assert metadata["mode"] == mode and metadata["oob_used_to_construct_control"] is False
                        assert metadata["source"] == ("temp_mlp" if source.endswith("temp_mlp") else "ppg_acc")
                        if offset != 0:
                            initialization = metadata["matched_initialization"]
                            assert initialization["mode"] == "real_c00"
                            assert initialization["oob_scores_used"] is False
                            assert initialization["untrained_identity_gate_only"] is True
                            lines = (directory/f"fold{fold}.log").read_text().splitlines()
                            training_end = [i for i,line in enumerate(lines) if "C10/10 " in line]
                            oob_start = [i for i,line in enumerate(lines) if "Building untouched claimed-target OOB cases for final reporting" in line]
                            assert len(training_end) == len(oob_start) == 1 and training_end[0] < oob_start[0], (directory,"training/OOB execution order")
                            execution_audits.append(dict(source=display,offset=offset,fold=fold,context=mode,log=str(directory/f"fold{fold}.log"),c10_log_line=training_end[0]+1,oob_construction_log_line=oob_start[0]+1,c10_completed_before_oob_construction=True))
                        if mode == "context_free":
                            assert len(metadata["train_fold_context_mean"]) == 4
                            assert metadata["train_context_rows_used_for_mean"] > 0
                        else:
                            assert metadata["context_features_permuted_jointly"] is True
                            assert metadata["maximum_forward_batch_rows"] == 512
                    expected_steps = [item["steps"] for item in real["calibration_history"]]
                    assert [item["steps"] for item in payload["calibration_history"]] == expected_steps
                    for horizon in HORIZONS:
                        data[(display,offset,fold,mode,horizon)] = macro(payload,horizon)
                        for seconds in DURATIONS:
                            for m in TOP_M:
                                actual = payload["calibration_epochs"][str(horizon)]["cells"][str(seconds)][str(m)]
                                ref = real["calibration_epochs"][str(horizon)]["cells"][str(seconds)][str(m)]
                                for block in [None, *actual["activity"]]:
                                    aa = actual["overall"] if block is None else actual["activity"][block]
                                    bb = ref["overall"] if block is None else ref["activity"][block]
                                    assert (aa["genuine_trials"],aa["impostor_trials"]) == (bb["genuine_trials"],bb["impostor_trials"])
                    c00 = load_checkpoint(directory/"checkpoints"/f"fold{fold}_C00.pt")
                    c10 = load_checkpoint(directory/"checkpoints"/f"fold{fold}_C10.pt")
                    assert set(c00["model_state"]) == set(reference_c00["model_state"])
                    assert all(torch.equal(value,reference_c00["model_state"][key]) for key,value in c00["model_state"].items()), (directory,"different C00 initialization")
                    for cp,horizon in [(c00,0),(c10,10)]:
                        assert cp["fold_id"] == fold and cp["calibration_epoch"] == horizon
                        assert cp["source_epoch"] == 50 and cp["source_variant"] == source
                        assert cp["source_model_state_sha256"] == source_cp["model_state_sha256"]
                        assert payload["calibration_epochs"][str(horizon)]["model_state_sha256"] == cp["model_state_sha256"]
                        assert cp["trainable_parameters"] == 13848
                        assert all(torch.equal(value,cp["model_state"][key]) for key,value in source_cp["model_state"].items()), (directory,"source changed")
                    assert len(c10["gate_optimizer_state"]["state"]) == 10
                    assert all(int(slot["step"].item()) == sum(expected_steps) for slot in c10["gate_optimizer_state"]["state"].values())
                    c00_delta = data[(display,offset,fold,mode,0)]-data[(display,offset,fold,"real",0)]
                    assert abs(c00_delta) <= 0.01+1e-12, (directory,"C00 evaluation parity",c00_delta)
                    audits.append(dict(source=display,offset=offset,fold=fold,context=mode,source_sha256=source_cp["model_state_sha256"],c00_sha256=c00["model_state_sha256"],c10_sha256=c10["model_state_sha256"],c00_delta_vs_real_percentage_points=c00_delta,matched_initialization=True,frozen_source_unchanged=True,trial_counts_match=True,optimizer_steps=sum(expected_steps)))
                    fold_rows.append(dict(source=display,offset=offset,fold=fold,condition=mode+" C10",eer_percent=data[(display,offset,fold,mode,10)]))
                fold_rows.append(dict(source=display,offset=offset,fold=fold,condition="Source C00",eer_percent=data[(display,offset,fold,"real",0)]))
            print(f"Validated {display}, offset={offset}", flush=True)

    if args.validate_only:
        filename = "validation_offsets_" + "_".join(str(offset) for offset in args.offsets) + "_sources_" + "_".join(SOURCES[source] for source in args.sources) + ".json"
        (OUTPUT/filename).write_text(json.dumps(dict(status="pass",seed_offsets=args.offsets,sources=args.sources,models_checked=len(audits),checks=audits),indent=2)+"\n")
        print(f"Validation only: {len(audits)} model conditions passed; no three-seed summary generated.",flush=True)
        return

    conditions = [("Source C00","real",0),("Context-free C10","context_free",10),("Shuffled-context C10","context_shuffled",10),("Real-context C10","real",10)]
    condition_rows=[]
    trajectory_rows=[]
    pair_rows=[]
    paired_folds=[]
    for display in SOURCES:
        for label,mode,horizon in conditions:
            values=[statistics.mean(data[(display,offset,fold,mode,horizon)] for fold in range(1,5)) for offset in ROOTS]
            condition_rows.append(dict(source=display,condition=label,**summarize(values)))
        for mode in MODES:
            for horizon in HORIZONS:
                values=[statistics.mean(data[(display,offset,fold,mode,horizon)] for fold in range(1,5)) for offset in ROOTS]
                trajectory_rows.append(dict(source=display,context=mode,calibration_epoch=horizon,**summarize(values)))
        for label,lm,lh,rm,rh in [
            ("Context-free C10 - Source C00","context_free",10,"real",0),
            ("Shuffled C10 - Source C00","context_shuffled",10,"real",0),
            ("Real C10 - Source C00","real",10,"real",0),
            ("Real C10 - Context-free C10","real",10,"context_free",10),
            ("Real C10 - Shuffled C10","real",10,"context_shuffled",10),
        ]:
            values=[]
            improving_folds=[]
            for offset in ROOTS:
                diffs=[data[(display,offset,fold,lm,lh)]-data[(display,offset,fold,rm,rh)] for fold in range(1,5)]
                values.append(statistics.mean(diffs))
                improving_folds.append(sum(value<0 for value in diffs))
                paired_folds.extend(dict(source=display,contrast=label,offset=offset,fold=fold,delta_eer_percentage_points=value) for fold,value in enumerate(diffs,1))
            pair_rows.append(dict(source=display,contrast=label,**summarize(values),improved_seeds=sum(value<0 for value in values),improved_folds_offset_0=improving_folds[0],improved_folds_offset_5000=improving_folds[1],improved_folds_offset_10000=improving_folds[2]))

    # Guard against accidentally mixing a legacy aggregate or another checkpoint.
    with (HERE/"results/FINAL_260802_THREE_SEED_SUMMARY/macro12_by_seed.csv").open() as handle:
        main_rows = {row["model"]: row for row in csv.DictReader(handle)}
    for row in condition_rows:
        if row["condition"] == "Real-context C10":
            main_name = "PPG+ACC+Gate" if row["source"] == "PPG+ACC" else "PPG+ACC+Temp MLP+Gate"
            for offset in ROOTS:
                assert abs(row[f"offset_{offset}_percent"]-float(main_rows[main_name][f"offset_{offset}_eer_percent"])) < 1e-10
    expected_seed0 = {("PPG+ACC","Context-free C10"):25.076, ("PPG+ACC","Shuffled-context C10"):24.378, ("PPG+ACC+Temp MLP","Context-free C10"):22.951, ("PPG+ACC+Temp MLP","Shuffled-context C10"):22.139}
    for row in condition_rows:
        key = (row["source"], row["condition"])
        if key in expected_seed0:
            assert abs(row["offset_0_percent"]-expected_seed0[key]) <= 0.0005

    write_csv("conditions_by_seed.csv",condition_rows)
    write_csv("fold_results.csv",fold_rows)
    write_csv("trajectory_by_seed.csv",trajectory_rows)
    write_csv("paired_delta_by_seed.csv",pair_rows)
    write_csv("paired_delta_by_fold.csv",paired_folds)
    progress = load_json(OUTPUT/"progress.json")
    assert progress["status"] == "complete" and progress["completed_jobs"] == 32
    assert len(execution_audits) == 32
    for name,expected in progress["code_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected, (name,"training code changed during execution")
    (OUTPUT/"execution_audit.json").write_text(json.dumps(dict(status="pass",new_jobs_checked=len(execution_audits),training_code_sha256=progress["code_sha256"],checks=execution_audits),indent=2)+"\n")
    (OUTPUT/"matched_audit.json").write_text(json.dumps(dict(status="pass",models_checked=len(audits),checks=audits),indent=2)+"\n")
    summary=dict(status="complete",created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),seed_offsets=list(ROOTS),aggregation="Within each seed, average 12 cell EERs and 4 folds equally; across seeds, mean and sample SD ddof=1. Paired differences are computed within seed.",source_c00="Existing real-context C00 of the same source and offset",conditions=condition_rows,paired_differences=pair_rows,limitations=["Three seeds are initialization repeats on fixed participant folds, not independent datasets.","Shuffled context is a joint row permutation within each extraction batch, not complete removal of all context information."])
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# Three-seed Gate context controls", "", "All EERs are percentages. ± is sample SD across three complete seed runs (ddof=1).", "", "| Source | Condition | Offset 0 | Offset 5000 | Offset 10000 | Mean ± seed SD |", "|---|---|---:|---:|---:|---:|"]
    for row in condition_rows:
        lines.append(f"| {row['source']} | {row['condition']} | {row['offset_0_percent']:.3f} | {row['offset_5000_percent']:.3f} | {row['offset_10000_percent']:.3f} | {row['mean_percent']:.3f} ± {row['seed_sample_sd_percentage_points']:.3f} |")
    lines += ["", "Paired differences use left minus right; negative values favor the left condition.", "", "| Source | Paired contrast | ΔEER ± seed SD (pp) | Improved seeds |", "|---|---|---:|---:|"]
    for row in pair_rows:
        lines.append(f"| {row['source']} | {row['contrast']} | {row['mean_percent']:+.3f} ± {row['seed_sample_sd_percentage_points']:.3f} | {row['improved_seeds']}/3 |")
    lines += ["", "Audit: 72 source/seed/fold/context model conditions validated, including exact matched C00 initialization, unchanged source parameters and buffers, matching trial counts and source/checkpoint hashes, and equal calibration updates.", "", "Seed-0 controls and all real-context results were reused; only the 32 missing fold controls at offsets 5000 and 10000 were trained.", ""]
    (OUTPUT/"README.md").write_text("\n".join(lines))
    print("\n".join(lines),flush=True)


if __name__ == "__main__":
    main()
