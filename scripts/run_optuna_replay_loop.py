from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from investment_backtest_lab.optuna_loop import (
    LoopPaths,
    append_loop_summary,
    build_failed_loop_summary_row,
    build_loop_summary_row,
    command_comparison,
    command_export,
    command_replay,
    command_review,
    command_search,
    command_shortlist,
    copy_cycle_snapshot,
    copy_final_snapshot,
    next_cycle_number,
    read_frame,
    run_command,
    write_final_shortlist,
    write_snapshot_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated Optuna search, shortlist, replay, and report cycles."
    )
    parser.add_argument("--config", default="configs/tw50_example.yaml")
    parser.add_argument("--family", default="tw50")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--study-name",
        default="tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508",
    )
    parser.add_argument(
        "--storage",
        default="reports/optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508.db",
    )
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--search-hours", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=50_000)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--final-candidates", type=int, default=5)
    parser.add_argument("--selector", default="hybrid_primary")
    parser.add_argument("--objective-profile", default="return_first")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--reviewer", default="Ian")
    parser.add_argument("--final-full-replay", action="store_true")
    parser.add_argument("--final-full-replay-only", action="store_true")
    parser.add_argument("--cost-stress-multiplier", type=float, default=2.0)
    parser.add_argument("--skip-cost-stress", action="store_true")
    args = parser.parse_args()
    if args.cost_stress_multiplier <= 0:
        raise SystemExit("--cost-stress-multiplier must be positive.")

    cwd = Path.cwd()
    paths = LoopPaths(Path(args.output_dir), args.study_name, args.family.lower())
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.final_full_replay_only:
        run_cycles(args, cwd=cwd, paths=paths)
    if args.final_full_replay or args.final_full_replay_only:
        run_final_full_replay(args, cwd=cwd, paths=paths)


def run_cycles(args, *, cwd: Path, paths: LoopPaths) -> None:
    next_cycle = next_cycle_number(paths.summary_csv)
    for offset in range(int(args.loops)):
        cycle = next_cycle + offset
        started_at = datetime.now(UTC)
        print(f"\n=== Loop cycle {cycle} started at {started_at.isoformat()} ===", flush=True)
        try:
            run_step(args, paths, cwd, cycle, "search", command_search(args))
            run_step(args, paths, cwd, cycle, "export", command_export(args))
            run_step(args, paths, cwd, cycle, "shortlist", command_shortlist(args))
            run_step(
                args,
                paths,
                cwd,
                cycle,
                "fast_replay",
                command_replay(args, shortlist_path=paths.shortlist_csv, full=False),
            )
            if not args.skip_cost_stress:
                run_step(
                    args,
                    paths,
                    cwd,
                    cycle,
                    "stress_2x_replay",
                    command_replay(
                        args,
                        shortlist_path=paths.shortlist_csv,
                        full=False,
                        cost_multiplier=args.cost_stress_multiplier,
                        report_suffix="stress_2x",
                    ),
                )
            run_step(args, paths, cwd, cycle, "comparison", command_comparison(args))
            run_step(args, paths, cwd, cycle, "review", command_review(args))
            row = build_loop_summary_row(
                cycle=cycle,
                study_name=args.study_name,
                started_at=started_at,
                ended_at=datetime.now(UTC),
                triage=read_frame(paths.triage_csv),
                shortlist=read_frame(paths.shortlist_csv),
                replay_ranking=read_frame(paths.replay_ranking_csv),
                stress_ranking=read_frame(paths.stress_replay_ranking_csv)
                if not args.skip_cost_stress
                else None,
            )
            summary = append_loop_summary(paths, row)
            snapshot = copy_cycle_snapshot(paths, cycle=cycle)
            index_path = write_snapshot_index(
                snapshot=snapshot,
                study_name=args.study_name,
                family=args.family,
                title=f"TW50 V5 Monthly + Weekly Trigger Loop Cycle {cycle}",
                cycle=cycle,
            )
            print(
                f"Cycle {cycle} status={row['loop_status']} "
                f"trials={row.get('completed_trials', '')} "
                f"top={row.get('replay_top_scenario_label', '')} "
                f"summary={paths.summary_html} snapshot={snapshot} index={index_path}",
                flush=True,
            )
            _ = summary
        except Exception as exc:
            failed = build_failed_loop_summary_row(
                cycle=cycle,
                study_name=args.study_name,
                started_at=started_at,
                ended_at=datetime.now(UTC),
                step="loop",
                error=str(exc),
            )
            append_loop_summary(paths, failed)
            print(f"Loop cycle {cycle} failed: {exc}", flush=True)
            raise SystemExit(1) from exc


def run_final_full_replay(args, *, cwd: Path, paths: LoopPaths) -> None:
    print("\n=== Building final full replay shortlist ===", flush=True)
    run_step(args, paths, cwd, 0, "export_final", command_export(args))
    summary = read_frame(paths.summary_csv)
    triage = read_frame(paths.triage_csv)
    final = write_final_shortlist(
        paths=paths,
        summary=summary,
        triage=triage,
        max_candidates=args.final_candidates,
    )
    if final.empty:
        raise SystemExit("No final candidates are available for full replay.")
    print(f"Final candidates: {len(final)} -> {paths.final_shortlist_csv}", flush=True)
    run_step(
        args,
        paths,
        cwd,
        0,
        "final_full_replay",
        command_replay(args, shortlist_path=paths.final_shortlist_csv, full=True),
    )
    if not args.skip_cost_stress:
        run_step(
            args,
            paths,
            cwd,
            0,
            "final_stress_2x_replay",
            command_replay(
                args,
                shortlist_path=paths.final_shortlist_csv,
                full=True,
                cost_multiplier=args.cost_stress_multiplier,
                report_suffix="stress_2x",
            ),
        )
    run_step(args, paths, cwd, 0, "final_comparison", command_comparison(args))
    run_step(args, paths, cwd, 0, "final_review", command_review(args))
    snapshot = copy_final_snapshot(paths)
    index_path = write_snapshot_index(
        snapshot=snapshot,
        study_name=args.study_name,
        family=args.family,
        title="TW50 V5 Monthly + Weekly Trigger Final Full Replay",
        cycle=None,
    )
    print(f"Final full replay completed. snapshot={snapshot} index={index_path}", flush=True)


def run_step(args, paths: LoopPaths, cwd: Path, cycle: int, step: str, command: list[str]) -> None:
    cycle_name = f"cycle_{cycle:04d}" if cycle else "final"
    log_path = paths.log_root / f"{cycle_name}_{step}.log"
    print(f"\n[{cycle_name}] {step}: {' '.join(command)}", flush=True)
    run_command(command, cwd=cwd, log_path=log_path)


if __name__ == "__main__":
    main()
