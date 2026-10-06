#!/bin/bash
set -euo pipefail

# Multi-agent in-loop orchestrator: 4 agent stages per iteration.
#
# Per iteration:
#
#   1. plan agent
#        → strategy_memo + eval_protocol + policy_protocol
#
#   2. formulation agent
#        design (codex)          → formulation code (fNNN.py)
#        verify (AST verifier)   → reject + retry design on contract violation
#                                  (max 3 retries; iter is skipped if all fail)
#
#   3. oracle agent
#        oracle search           → full Sobol grid + zero-pinned ablations
#                                  (results.json + raw.npz)
#        oracle analysis (codex) → expressivity report (markdown)
#
#   4. policy agent
#        policy training         → step3_bandit.train 20 epochs
#                                  (test computed but agent-isolated)
#        policy analysis (codex) → learnability report (markdown)
#
#   record → history.jsonl entry with summaries + derived trend metrics
#
# No post-loop sweep: best-by-val selection happens via final_analysis.py.
#
# Usage:
#   bash experiments/cuerator/orchestrate.sh [--start-iter N] [--run-name NAME]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EXPERIMENTS_DIR")"
cd "$PROJECT_DIR"

source env.sh

resolve_conda_python() {
    local env_name="$1"
    local conda_bin
    local conda_root
    local python_path

    conda_bin="$(command -v conda)"
    conda_root="$(dirname "$(dirname "$conda_bin")")"

    if [[ "$env_name" == "base" ]]; then
        python_path="$conda_root/bin/python3"
    else
        python_path="$conda_root/envs/$env_name/bin/python3"
    fi

    if [[ ! -x "$python_path" ]]; then
        echo "Missing python3 for conda env '$env_name': $python_path" >&2
        exit 1
    fi

    printf '%s\n' "$python_path"
}

CONDA_ENV="${CONDA_ENV:-cuerator}"
export CONDA_NO_PLUGINS=true
CONDA_PY="$(resolve_conda_python "$CONDA_ENV")"
PY="conda run -n $CONDA_ENV --no-capture-output env OVAVEL_ROOT=$OVAVEL_ROOT $CONDA_PY"
HELPER="$PY -m experiments.cuerator.helpers"

CONFIG="experiments/cuerator/config.yaml"
PLAN_TEMPLATE="experiments/cuerator/prompts/plan.md"
DESIGN_TEMPLATE="experiments/cuerator/prompts/design.md"
ORACLE_TEMPLATE="experiments/cuerator/prompts/oracle.md"
POLICY_TEMPLATE="experiments/cuerator/prompts/policy.md"
WRAP_AGENT="$SCRIPT_DIR/wrap_agent.sh"

MAX_ITER=$($HELPER config-get --config $CONFIG --key search.max_iterations --default 50)
MAX_RETRIES=$($HELPER config-get --config $CONFIG --key search.max_retries --default 3)
NUM_PARAMS=$($HELPER config-get --config $CONFIG --key search.num_params --default 5)
# S1/S2 verifier size limits — rendered into the Design prompt AND passed to the
# verifier (helpers reads the same keys) so prompt and AST check stay in sync.
MAX_HELPERS=$($HELPER config-get --config $CONFIG --key verifier.max_helpers --default 5)
MAX_HELPER_STMTS=$($HELPER config-get --config $CONFIG --key verifier.max_helper_stmts --default 8)
MAX_HELPER_TOTAL=$($HELPER config-get --config $CONFIG --key verifier.max_helper_total --default 16)
MAX_MAIN_STMTS=$($HELPER config-get --config $CONFIG --key verifier.max_main_stmts --default 18)
MAX_FUNCTIONS=$((MAX_HELPERS + 2))
AGENT_MODEL=$($HELPER config-get --config $CONFIG --key backend.model)
AGENT_MODEL_FLAG=""
if [[ -n "$AGENT_MODEL" ]]; then
    AGENT_MODEL_FLAG="-m $AGENT_MODEL"
fi
REASONING_EFFORT=$($HELPER config-get --config $CONFIG --key backend.reasoning_effort --default medium)
AGENT_REASONING_FLAG="-c model_reasoning_effort=\"$REASONING_EFFORT\""

SESSIONS_ROOT="runs/cuerator"

START_ITER=1
RUN_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-iter) START_ITER="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="$(openssl rand -hex 3)"
    while [[ -e "$SESSIONS_ROOT/$RUN_NAME" ]]; do
        RUN_NAME="$(openssl rand -hex 3)"
    done
fi

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "$RUN_NAME" == "latest" ]]; then
    echo "Invalid --run-name '$RUN_NAME'."
    exit 1
fi

SESSION_DIR="$SESSIONS_ROOT/$RUN_NAME"
AGENT_CTX="$SESSION_DIR/agent_context"
FORMULATIONS_DIR="$AGENT_CTX/formulations"
RESULTS_DIR="$AGENT_CTX/results"
STRATEGY_MEMO_DIR="$AGENT_CTX/strategy_memo"
EVAL_PROTOCOL_DIR="$AGENT_CTX/eval_protocol"
POLICY_PROTOCOL_DIR="$AGENT_CTX/policy_protocol"
ORACLE_REPORTS_DIR="$AGENT_CTX/oracle_reports"
POLICY_REPORTS_DIR="$AGENT_CTX/policy_reports"
POLICY_RESULTS_DIR="$AGENT_CTX/policy_results"
ORACLE_SEARCH_RAW_DIR="$SESSION_DIR/oracle_search_raw"
POLICY_RAW_DIR="$SESSION_DIR/policy_raw"
POLICY_RUNS_DIR="$SESSION_DIR/policy_runs"
HISTORY="$AGENT_CTX/history.jsonl"
LAST_FAILURE_FILE="$SESSION_DIR/last_failure.txt"
DESIGN_TMP_ROOT="$SESSION_DIR/tmp"
ARCHIVE_DIR="$SESSION_DIR/archived_formulations"
ARCHIVE_MEMO_DIR="$SESSION_DIR/archived_strategy_memos"
ARCHIVE_EVAL_PROTOCOL_DIR="$SESSION_DIR/archived_eval_protocols"
ARCHIVE_POLICY_PROTOCOL_DIR="$SESSION_DIR/archived_policy_protocols"
ARCHIVE_ORACLE_REPORT_DIR="$SESSION_DIR/archived_oracle_reports"
ARCHIVE_POLICY_REPORT_DIR="$SESSION_DIR/archived_policy_reports"
STATE="$SESSION_DIR/state.json"
AGENT_REASONING_LOG="$SESSION_DIR/agent_reasoning.log"
ORCHESTRATE_LOG="$SESSION_DIR/orchestrate.log"
FULL_HISTORY="$SESSION_DIR/history.jsonl"

RUN_EXISTS=false
if [[ -d "$SESSION_DIR" ]]; then
    RUN_EXISTS=true
fi

mkdir -p "$FORMULATIONS_DIR" "$RESULTS_DIR" "$STRATEGY_MEMO_DIR" "$EVAL_PROTOCOL_DIR" \
         "$POLICY_PROTOCOL_DIR" "$ORACLE_REPORTS_DIR" "$POLICY_REPORTS_DIR" \
         "$POLICY_RESULTS_DIR" "$ORACLE_SEARCH_RAW_DIR" "$POLICY_RAW_DIR" "$POLICY_RUNS_DIR" \
         "$ARCHIVE_DIR" "$ARCHIVE_MEMO_DIR" "$ARCHIVE_EVAL_PROTOCOL_DIR" \
         "$ARCHIVE_POLICY_PROTOCOL_DIR" "$ARCHIVE_ORACLE_REPORT_DIR" "$ARCHIVE_POLICY_REPORT_DIR" \
         "$DESIGN_TMP_ROOT"
touch "$HISTORY" "$AGENT_REASONING_LOG" "$ORCHESTRATE_LOG" "$FULL_HISTORY"

# Snapshot config into the session dir so later edits to the shared config.yaml
# don't leak into already-running sessions. Resume preserves the original snapshot.
SESSION_CONFIG="$SESSION_DIR/config.yaml"
if [[ ! -f "$SESSION_CONFIG" ]]; then
    cp "$CONFIG" "$SESSION_CONFIG"
fi
CONFIG="$SESSION_CONFIG"

exec > >(tee -a "$ORCHESTRATE_LOG") 2>&1

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

format_elapsed() {
    local total="$1"
    printf '%02d:%02d:%02d (%ss)' \
        $((total / 3600)) \
        $(((total % 3600) / 60)) \
        $((total % 60)) \
        "$total"
}

log_end_and_elapsed() {
    local prefix="$1"
    local start_epoch="$2"
    local end_epoch
    end_epoch=$(date '+%s')
    echo "  $prefix end: $(timestamp)"
    echo "  $prefix elapsed: $(format_elapsed "$((end_epoch - start_epoch))")"
}

if [[ ! -f "$STATE" ]]; then
    echo '{"iteration": 0, "best_train_oracle_frame_acc": 0, "best_val_frame_acc": -1}' > "$STATE"
fi

RUN_START_EPOCH=$(date '+%s')

echo "=== CueRator ==="
echo "  Run start: $(timestamp)"
echo "  Model: ${AGENT_MODEL:-default}"
echo "  Run mode: $([[ "$RUN_EXISTS" == "true" ]] && echo resume || echo new)"
echo "  Run: $RUN_NAME"
echo "  Logs: $SESSION_DIR"
echo "  Iterations: $START_ITER to $MAX_ITER"
echo ""

archive_file() {
    local file="$1"
    local archive_dir="$2"
    local reason="$3"
    local ext="${4:-}"
    local base timestamp_str archived
    [[ -f "$file" ]] || return 0
    timestamp_str=$(date '+%Y%m%d_%H%M%S')
    if [[ -n "$ext" ]]; then
        base="$(basename "$file" ".$ext")"
        archived="$archive_dir/${base}__${reason}_${timestamp_str}.$ext"
    else
        base="$(basename "$file")"
        archived="$archive_dir/${base}__${reason}_${timestamp_str}"
    fi
    mv "$file" "$archived"
    echo "  Archived to $(basename "$archive_dir")/$(basename "$archived")"
}

run_agent_with_reasoning_log() {
    local role="$1"
    local prompt="$2"
    local label="$3"
    local transcript exit_code=0 start_epoch start_ts end_epoch end_ts

    transcript=$(mktemp)
    start_epoch=$(date '+%s')
    start_ts=$(timestamp)

    if printf '%s\n' "$prompt" | "$WRAP_AGENT" "$role" "$SESSION_DIR" "$(dirname "$CONDA_PY")" -- codex exec \
        --sandbox workspace-write \
        $AGENT_MODEL_FLAG \
        $AGENT_REASONING_FLAG \
        -C /work \
        --skip-git-repo-check \
        - 2>&1 | tee "$transcript"; then
        exit_code=0
    else
        exit_code=$?
    fi

    if grep -q "You've hit your usage limit" "$transcript" 2>/dev/null; then
        local reset_str wait_secs target_epoch now_epoch
        wait_secs=600
        reset_str=$(grep -o 'try again at [0-9]*:[0-9]* [AP]M' "$transcript" | head -1 | sed 's/try again at //')
        if [[ -n "$reset_str" ]]; then
            target_epoch=$(date -d "$reset_str" '+%s' 2>/dev/null || echo "")
            now_epoch=$(date '+%s')
            if [[ -n "$target_epoch" ]] && [[ "$target_epoch" -gt "$now_epoch" ]]; then
                wait_secs=$(( target_epoch - now_epoch + 60 ))
            fi
        fi
        LAST_AGENT_ELAPSED=$(($(date '+%s') - start_epoch))
        echo "  [QUOTA] Usage limit hit. Sleeping ${wait_secs}s (reset: ${reset_str:-unknown})."
        rm -f "$transcript"
        sleep "$wait_secs"
        return 99
    fi

    end_epoch=$(date '+%s')
    end_ts=$(timestamp)
    LAST_AGENT_ELAPSED=$((end_epoch - start_epoch))
    {
        echo "=== $label ==="
        echo "start: $start_ts"
        echo "end: $end_ts"
        echo "elapsed: $(format_elapsed "$LAST_AGENT_ELAPSED")"
        echo ""
    } >> "$AGENT_REASONING_LOG"
    $PY -m experiments.cuerator.helpers extract-reasoning \
        --input "$transcript" \
        --output "$AGENT_REASONING_LOG" || true
    rm -f "$transcript"
    return "$exit_code"
}

LAST_AGENT_ELAPSED=0

for iter in $(seq "$START_ITER" "$MAX_ITER"); do
    ITER_STR=$(printf "%03d" "$iter")
    FORM_FILE="$FORMULATIONS_DIR/f${ITER_STR}.py"
    MEMO_FILE="$STRATEGY_MEMO_DIR/iter_${ITER_STR}.md"
    EVAL_PROTOCOL_FILE="$EVAL_PROTOCOL_DIR/iter_${ITER_STR}.json"
    POLICY_PROTOCOL_FILE="$POLICY_PROTOCOL_DIR/iter_${ITER_STR}.json"
    ORACLE_REPORT_FILE="$ORACLE_REPORTS_DIR/iter_${ITER_STR}.md"
    POLICY_REPORT_FILE="$POLICY_REPORTS_DIR/iter_${ITER_STR}.md"
    RESULT_FILE="$RESULTS_DIR/iter_${ITER_STR}.json"
    POLICY_RESULT_FILE="$POLICY_RESULTS_DIR/iter_${ITER_STR}.json"
    ORACLE_RAW_FILE="$ORACLE_SEARCH_RAW_DIR/iter_${ITER_STR}.npz"
    POLICY_RAW_FILE="$POLICY_RAW_DIR/iter_${ITER_STR}.npz"
    POLICY_RUN_DIR="$POLICY_RUNS_DIR/iter_${ITER_STR}"
    ITER_START_EPOCH=$(date '+%s')

    echo "--- Iteration $iter / $MAX_ITER ---"
    echo "  Iteration start: $(timestamp)"

    if [[ -f "$RESULT_FILE" ]] && [[ -f "$POLICY_RESULT_FILE" ]]; then
        if $PY -c "import json,sys; d=json.load(open('$RESULT_FILE')); sys.exit(0 if 'best_train' in d else 1)" 2>/dev/null; then
            echo "  Result + policy_result exist, skipping."
            log_end_and_elapsed "Iteration" "$ITER_START_EPOCH"
            echo ""
            continue
        fi
    fi

    retry=0
    success=false
    REUSE_EXISTING_FORMULATION=false
    PLAN_SECONDS_TOTAL=0
    DESIGN_SECONDS_TOTAL=0
    ORACLE_SEARCH_SECONDS_TOTAL=0
    ORACLE_AGENT_SECONDS_TOTAL=0
    POLICY_TRAIN_SECONDS_TOTAL=0
    POLICY_SECONDS_TOTAL=0
    LAST_FAILURE_REASON=""
    rm -f "$LAST_FAILURE_FILE"

    if [[ -s "$FORM_FILE" ]] && [[ -s "$MEMO_FILE" ]] && [[ -s "$EVAL_PROTOCOL_FILE" ]] && [[ -s "$POLICY_PROTOCOL_FILE" ]]; then
        REUSE_EXISTING_FORMULATION=true
        echo "  Found existing formulation + memo + protocols; will reuse on first attempt."
    fi

    while [[ $retry -lt $MAX_RETRIES ]] && [[ "$success" != "true" ]]; do
        if [[ "$REUSE_EXISTING_FORMULATION" == "true" ]] && [[ $retry -eq 0 ]]; then
            echo "  Reusing existing formulation: $(basename "$FORM_FILE")"
            REUSE_EXISTING_FORMULATION=false
        else
            ARCHIVE_REASON="stale"
            if [[ $retry -gt 0 ]]; then
                ARCHIVE_REASON="retry_$(printf '%02d' "$retry")"
            fi
            archive_file "$FORM_FILE" "$ARCHIVE_DIR" "$ARCHIVE_REASON" "py"

            if [[ -f "$MEMO_FILE" ]] && [[ $retry -gt 0 ]]; then
                PRIOR_MEMO_SNAPSHOT=$(mktemp)
                cp "$MEMO_FILE" "$PRIOR_MEMO_SNAPSHOT"
            fi
            archive_file "$MEMO_FILE" "$ARCHIVE_MEMO_DIR" "$ARCHIVE_REASON" "md"
            archive_file "$EVAL_PROTOCOL_FILE" "$ARCHIVE_EVAL_PROTOCOL_DIR" "$ARCHIVE_REASON" "json"
            archive_file "$POLICY_PROTOCOL_FILE" "$ARCHIVE_POLICY_PROTOCOL_DIR" "$ARCHIVE_REASON" "json"

            if [[ $retry -gt 0 ]]; then
                {
                    echo "=== prior strategy memo (iter ${ITER_STR}, attempt ${retry}) ==="
                    if [[ -n "${PRIOR_MEMO_SNAPSHOT:-}" ]] && [[ -f "${PRIOR_MEMO_SNAPSHOT:-}" ]]; then
                        cat "$PRIOR_MEMO_SNAPSHOT"
                    else
                        echo "(prior memo not available)"
                    fi
                    echo ""
                    echo "=== failure reason (attempt ${retry}) ==="
                    echo "${LAST_FAILURE_REASON:-unknown}"
                } > "$LAST_FAILURE_FILE"
                rm -f "${PRIOR_MEMO_SNAPSHOT:-}"
                unset PRIOR_MEMO_SNAPSHOT
            fi

            # ─── Stage 1: Plan agent ────────────────────────────────────
            # Writes strategy_memo + eval_protocol + policy_protocol.
            PLAN_PROMPT=$(sed \
                -e "s|{{output_file_memo}}|iter_${ITER_STR}.md|g" \
                -e "s|{{output_file_eval_protocol}}|iter_${ITER_STR}.json|g" \
                -e "s|{{output_file_policy_protocol}}|iter_${ITER_STR}.json|g" \
                "$PLAN_TEMPLATE")

            echo "  Running Plan agent (attempt $((retry+1))/$MAX_RETRIES)..."
            export PLAN_LAST_FAILURE=""
            if [[ $retry -gt 0 ]] && [[ -f "$LAST_FAILURE_FILE" ]]; then
                export PLAN_LAST_FAILURE="$LAST_FAILURE_FILE"
            fi
            if run_agent_with_reasoning_log "plan" "$PLAN_PROMPT" "iter_${ITER_STR} plan attempt_$((retry+1))"; then
                PLAN_SECONDS_TOTAL=$((PLAN_SECONDS_TOTAL + LAST_AGENT_ELAPSED))
                echo "  Plan completed (elapsed: $(format_elapsed "$LAST_AGENT_ELAPSED"))."
            else
                AGENT_EXIT=$?
                PLAN_SECONDS_TOTAL=$((PLAN_SECONDS_TOTAL + LAST_AGENT_ELAPSED))
                unset PLAN_LAST_FAILURE
                if [[ "$AGENT_EXIT" -eq 99 ]]; then
                    echo "  Retrying plan after quota wait..."
                    continue
                fi
                LAST_FAILURE_REASON="Plan agent exited with code $AGENT_EXIT"
                echo "  Plan failed: $LAST_FAILURE_REASON"
                retry=$((retry + 1))
                continue
            fi
            unset PLAN_LAST_FAILURE

            if [[ ! -s "$MEMO_FILE" ]]; then
                LAST_FAILURE_REASON="Plan did not create strategy_memo/iter_${ITER_STR}.md"
                echo "  Plan output missing memo: $LAST_FAILURE_REASON"
                retry=$((retry + 1)); continue
            fi
            if [[ ! -s "$EVAL_PROTOCOL_FILE" ]]; then
                LAST_FAILURE_REASON="Plan did not create eval_protocol/iter_${ITER_STR}.json"
                echo "  Plan output missing eval_protocol: $LAST_FAILURE_REASON"
                retry=$((retry + 1)); continue
            fi
            if [[ ! -s "$POLICY_PROTOCOL_FILE" ]]; then
                LAST_FAILURE_REASON="Plan did not create policy_protocol/iter_${ITER_STR}.json"
                echo "  Plan output missing policy_protocol: $LAST_FAILURE_REASON"
                retry=$((retry + 1)); continue
            fi
            EVAL_PROTOCOL_VERIFY=$($HELPER verify-protocol --file "$EVAL_PROTOCOL_FILE")
            if [[ "$EVAL_PROTOCOL_VERIFY" != "ok" ]]; then
                LAST_FAILURE_REASON="eval_protocol verify failed: $EVAL_PROTOCOL_VERIFY"
                echo "  Eval protocol verify failed: $EVAL_PROTOCOL_VERIFY"
                retry=$((retry + 1)); continue
            fi
            POLICY_PROTOCOL_VERIFY=$($HELPER verify-policy-protocol --file "$POLICY_PROTOCOL_FILE")
            if [[ "$POLICY_PROTOCOL_VERIFY" != "ok" ]]; then
                LAST_FAILURE_REASON="policy_protocol verify failed: $POLICY_PROTOCOL_VERIFY"
                echo "  Policy protocol verify failed: $POLICY_PROTOCOL_VERIFY"
                retry=$((retry + 1)); continue
            fi

            # ─── Stage 2: Formulation agent / design ───────────────────
            # Writes formulation code (fNNN.py) under design's tmp output dir.
            # Verify (Stage 2's second component) runs below after the file moves
            # into place; on rejection, this whole retry block runs again.
            DESIGN_OUTPUT_DIR_ITER="$DESIGN_TMP_ROOT/design_iter_${ITER_STR}_attempt_$((retry+1))"
            rm -rf "$DESIGN_OUTPUT_DIR_ITER"
            mkdir -p "$DESIGN_OUTPUT_DIR_ITER"

            DESIGN_PROMPT=$(sed \
                -e "s|{{output_file}}|f${ITER_STR}.py|g" \
                -e "s|{{num_params}}|${NUM_PARAMS}|g" \
                -e "s|{{max_functions}}|${MAX_FUNCTIONS}|g" \
                -e "s|{{max_helpers}}|${MAX_HELPERS}|g" \
                -e "s|{{max_helper_stmts}}|${MAX_HELPER_STMTS}|g" \
                -e "s|{{max_helper_total}}|${MAX_HELPER_TOTAL}|g" \
                -e "s|{{max_main_stmts}}|${MAX_MAIN_STMTS}|g" \
                "$DESIGN_TEMPLATE")

            echo "  Running Design agent (attempt $((retry+1))/$MAX_RETRIES)..."
            export DESIGN_CURRENT_MEMO="$MEMO_FILE"
            export DESIGN_OUTPUT_DIR="$DESIGN_OUTPUT_DIR_ITER"
            if run_agent_with_reasoning_log "design" "$DESIGN_PROMPT" "iter_${ITER_STR} design attempt_$((retry+1))"; then
                DESIGN_SECONDS_TOTAL=$((DESIGN_SECONDS_TOTAL + LAST_AGENT_ELAPSED))
                echo "  Design completed (elapsed: $(format_elapsed "$LAST_AGENT_ELAPSED"))."
            else
                AGENT_EXIT=$?
                DESIGN_SECONDS_TOTAL=$((DESIGN_SECONDS_TOTAL + LAST_AGENT_ELAPSED))
                unset DESIGN_CURRENT_MEMO DESIGN_OUTPUT_DIR
                if [[ "$AGENT_EXIT" -eq 99 ]]; then
                    echo "  Retrying design after quota wait..."
                    continue
                fi
                LAST_FAILURE_REASON="Design agent exited with code $AGENT_EXIT"
                retry=$((retry + 1)); continue
            fi
            unset DESIGN_CURRENT_MEMO DESIGN_OUTPUT_DIR

            DESIGN_OUTPUT_FILE="$DESIGN_OUTPUT_DIR_ITER/f${ITER_STR}.py"
            if [[ ! -s "$DESIGN_OUTPUT_FILE" ]]; then
                LAST_FAILURE_REASON="Design did not create output/f${ITER_STR}.py"
                retry=$((retry + 1)); continue
            fi
            mv "$DESIGN_OUTPUT_FILE" "$FORM_FILE"
            rm -rf "$DESIGN_OUTPUT_DIR_ITER"
        fi

        if [[ ! -s "$FORM_FILE" ]]; then
            LAST_FAILURE_REASON="formulation file empty/missing"
            retry=$((retry + 1)); continue
        fi

        # ─── Stage 2: Formulation agent / verify ───────────────────
        # AST contract verifier (function count, statement budget, no
        # control flow, allowed constants, ...). Rejects → retry design.
        VERIFY=$($HELPER verify-formulation --file "$FORM_FILE" --config "$CONFIG")
        if [[ "$VERIFY" != "ok" ]]; then
            LAST_FAILURE_REASON="verify rejected: $VERIFY"
            echo "  Formulation verify failed: $VERIFY"
            retry=$((retry + 1)); continue
        fi

        # ─── Stage 3: Expressivity agent / oracle search ───────────
        # Sobol grid + zero-pinned ablations → results.json + raw.npz.
        # The verify step above is the sole gate; evaluate.py assumes the
        # formulation has already passed contract verification.
        echo "  Evaluating (oracle search + ablations)..."
        EVAL_START_EPOCH=$(date '+%s')
        EVAL_TRANSCRIPT=$(mktemp)
        if $PY -m experiments.cuerator.evaluate \
            --formulation "$FORM_FILE" \
            --config "$CONFIG" \
            --output "$RESULT_FILE" \
            --protocol "$EVAL_PROTOCOL_FILE" \
            --raw "$ORACLE_RAW_FILE" 2>&1 | tee "$EVAL_TRANSCRIPT"; then
            success=true
        else
            EVAL_REASON=$(grep -E "REJECTED|Error|error" "$EVAL_TRANSCRIPT" | head -3 | tr '\n' '; ' || true)
            LAST_FAILURE_REASON="evaluation failed: ${EVAL_REASON:-unknown}"
            echo "  Evaluation failed (attempt $((retry+1)))."
            retry=$((retry + 1))
        fi
        rm -f "$EVAL_TRANSCRIPT"
        ORACLE_SEARCH_SECONDS_TOTAL=$((ORACLE_SEARCH_SECONDS_TOTAL + $(date '+%s') - EVAL_START_EPOCH))
        log_end_and_elapsed "Evaluation" "$EVAL_START_EPOCH"
    done

    if [[ "$success" != "true" ]]; then
        echo "  FAILED after $MAX_RETRIES retries. Skipping."
        echo "{\"iter\": $iter, \"error\": \"failed after $MAX_RETRIES retries\"}" >> "$HISTORY"
        $HELPER record-failure --state "$STATE" --iter "$iter"
        rm -f "$LAST_FAILURE_FILE"
        log_end_and_elapsed "Iteration" "$ITER_START_EPOCH"
        echo ""
        continue
    fi

    rm -f "$LAST_FAILURE_FILE"

    # Compute absolute paths once for both Evaluator and Policy agents.
    RESULT_ABS="$(realpath "$RESULT_FILE")"
    FORM_ABS="$(realpath "$FORM_FILE")"
    MEMO_ABS="$(realpath "$MEMO_FILE")"
    EVAL_PROTOCOL_ABS="$(realpath "$EVAL_PROTOCOL_FILE")"
    POLICY_PROTOCOL_ABS="$(realpath "$POLICY_PROTOCOL_FILE")"
    ORACLE_RAW_ABS="$(realpath "$ORACLE_RAW_FILE")"
    ORACLE_REPORT_ABS="$(realpath -m "$ORACLE_REPORT_FILE")"
    POLICY_REPORT_ABS="$(realpath -m "$POLICY_REPORT_FILE")"
    POLICY_RESULT_ABS="$(realpath -m "$POLICY_RESULT_FILE")"
    POLICY_RAW_ABS="$(realpath -m "$POLICY_RAW_FILE")"

    # ─── Stage 3: Expressivity agent / oracle analysis (non-fatal) ─
    # Reads results.json + raw.npz; writes expressivity report.
    ORACLE_PROMPT=$(cat "$ORACLE_TEMPLATE")
    echo "  Running Oracle Analysis agent..."
    archive_file "$ORACLE_REPORT_FILE" "$ARCHIVE_ORACLE_REPORT_DIR" "stale" "md"
    export ORACLE_RESULT_FILE="$RESULT_ABS" \
           ORACLE_FORMULATION_FILE="$FORM_ABS" \
           ORACLE_MEMO_FILE="$MEMO_ABS" \
           EVAL_PROTOCOL_FILE="$EVAL_PROTOCOL_ABS" \
           ORACLE_RAW_FILE="$ORACLE_RAW_ABS" \
           ORACLE_REPORT_FILE="$ORACLE_REPORT_ABS"
    if run_agent_with_reasoning_log "oracle" "$ORACLE_PROMPT" "iter_${ITER_STR} oracle_analysis"; then
        ORACLE_AGENT_SECONDS_TOTAL=$((ORACLE_AGENT_SECONDS_TOTAL + LAST_AGENT_ELAPSED))
        if [[ -s "$ORACLE_REPORT_FILE" ]]; then
            echo "  Oracle analysis report saved (elapsed: $(format_elapsed "$LAST_AGENT_ELAPSED"))."
        else
            echo "  Oracle analysis returned 0 but report is empty (non-fatal)."
        fi
    else
        AGENT_EXIT=$?
        ORACLE_AGENT_SECONDS_TOTAL=$((ORACLE_AGENT_SECONDS_TOTAL + LAST_AGENT_ELAPSED))
        if [[ "$AGENT_EXIT" -eq 99 ]]; then
            echo "  Oracle analysis hit quota; report missing this iter (non-fatal)."
        else
            echo "  Oracle analysis failed (exit $AGENT_EXIT, non-fatal)."
        fi
    fi
    unset ORACLE_RESULT_FILE ORACLE_FORMULATION_FILE ORACLE_MEMO_FILE EVAL_PROTOCOL_FILE ORACLE_RAW_FILE ORACLE_REPORT_FILE

    # ─── Stage 4: Policy agent / training ──────────────────────────
    # 20-epoch REINFORCE; learns a sample-conditioned transformer policy
    # that outputs a per-sample (P,) parameter vector. Test is computed
    # but kept out of the agent's view (test isolation).
    echo "  Running policy training (20 epochs)..."
    POLICY_TRAIN_START=$(date '+%s')
    POLICY_TRAIN_TRANSCRIPT=$(mktemp)
    rm -rf "$POLICY_RUN_DIR"
    if $PY -m experiments.cuerator.run_policy \
        --formulation "$FORM_FILE" \
        --config "$CONFIG" \
        --policy-run-dir "$POLICY_RUN_DIR" \
        --policy-results "$POLICY_RESULT_FILE" \
        --policy-raw "$POLICY_RAW_FILE" 2>&1 | tee "$POLICY_TRAIN_TRANSCRIPT"; then
        POLICY_TRAIN_OK=true
    else
        POLICY_TRAIN_OK=false
        echo "  Policy training FAILED (non-fatal — policy analysis will be skipped)."
    fi
    rm -f "$POLICY_TRAIN_TRANSCRIPT"
    POLICY_TRAIN_SECONDS_TOTAL=$(($(date '+%s') - POLICY_TRAIN_START))
    log_end_and_elapsed "Policy training" "$POLICY_TRAIN_START"

    # ─── Stage 4: Policy agent / analysis (non-fatal) ──────────────
    # Reads policy training results + oracle context; writes learnability
    # report. Skipped if policy training itself failed.
    if [[ "$POLICY_TRAIN_OK" == "true" ]] && [[ -s "$POLICY_RESULT_FILE" ]] && [[ -s "$POLICY_RAW_FILE" ]]; then
        POLICY_PROMPT=$(cat "$POLICY_TEMPLATE")
        echo "  Running Policy Analysis agent..."
        archive_file "$POLICY_REPORT_FILE" "$ARCHIVE_POLICY_REPORT_DIR" "stale" "md"
        export POL_RESULT_FILE="$RESULT_ABS" \
               POL_RAW_FILE="$ORACLE_RAW_ABS" \
               POL_POLICY_RESULT="$POLICY_RESULT_ABS" \
               POL_POLICY_RAW="$POLICY_RAW_ABS" \
               POL_FORMULATION_FILE="$FORM_ABS" \
               POL_MEMO_FILE="$MEMO_ABS" \
               POL_EVAL_PROTOCOL="$EVAL_PROTOCOL_ABS" \
               POL_POLICY_PROTOCOL="$POLICY_PROTOCOL_ABS" \
               POL_ORACLE_REPORT="$ORACLE_REPORT_ABS" \
               POL_REPORT_FILE="$POLICY_REPORT_ABS"
        if run_agent_with_reasoning_log "policy" "$POLICY_PROMPT" "iter_${ITER_STR} policy_analysis"; then
            POLICY_SECONDS_TOTAL=$((POLICY_SECONDS_TOTAL + LAST_AGENT_ELAPSED))
            if [[ -s "$POLICY_REPORT_FILE" ]]; then
                echo "  Policy analysis report saved (elapsed: $(format_elapsed "$LAST_AGENT_ELAPSED"))."
            else
                echo "  Policy analysis returned 0 but report is empty (non-fatal)."
            fi
        else
            AGENT_EXIT=$?
            POLICY_SECONDS_TOTAL=$((POLICY_SECONDS_TOTAL + LAST_AGENT_ELAPSED))
            if [[ "$AGENT_EXIT" -eq 99 ]]; then
                echo "  Policy analysis hit quota; report missing this iter (non-fatal)."
            else
                echo "  Policy analysis failed (exit $AGENT_EXIT, non-fatal)."
            fi
        fi
        unset POL_RESULT_FILE POL_RAW_FILE POL_POLICY_RESULT POL_POLICY_RAW \
              POL_FORMULATION_FILE POL_MEMO_FILE POL_EVAL_PROTOCOL POL_POLICY_PROTOCOL \
              POL_ORACLE_REPORT POL_REPORT_FILE
    else
        echo "  Skipping policy analysis (training did not complete cleanly)."
    fi

    SUMMARY=$($HELPER record-result \
        --result "$RESULT_FILE" \
        --memo "$MEMO_FILE" \
        --oracle-report "$ORACLE_REPORT_ABS" \
        --policy-report "$POLICY_REPORT_ABS" \
        --policy-result "$POLICY_RESULT_FILE" \
        --history "$HISTORY" \
        --state "$STATE" \
        --iter "$iter" \
        --full-history "$FULL_HISTORY" \
        --plan-seconds "$PLAN_SECONDS_TOTAL" \
        --design-seconds "$DESIGN_SECONDS_TOTAL" \
        --oracle-search-seconds "$ORACLE_SEARCH_SECONDS_TOTAL" \
        --oracle-agent-seconds "$ORACLE_AGENT_SECONDS_TOTAL" \
        --policy-train-seconds "$POLICY_TRAIN_SECONDS_TOTAL" \
        --policy-seconds "$POLICY_SECONDS_TOTAL")
    echo "  Done: $SUMMARY"
    echo "  Plan: $(format_elapsed "$PLAN_SECONDS_TOTAL")  Design: $(format_elapsed "$DESIGN_SECONDS_TOTAL")  Oracle: $(format_elapsed "$ORACLE_SEARCH_SECONDS_TOTAL")  Evaluator: $(format_elapsed "$ORACLE_AGENT_SECONDS_TOTAL")  PolicyTrain: $(format_elapsed "$POLICY_TRAIN_SECONDS_TOTAL")  Policy: $(format_elapsed "$POLICY_SECONDS_TOTAL")"

    log_end_and_elapsed "Iteration" "$ITER_START_EPOCH"

    # Cleanup per-iter raw arrays - agents have consumed them; safe to drop now.
    rm -f "$ORACLE_RAW_ABS" "$POLICY_RAW_FILE"

    echo ""
done

log_end_and_elapsed "Search loop" "$RUN_START_EPOCH"
echo "=== Search loop complete ==="
echo ""

echo "=== Final analysis (best val → test mapping) ==="
FINAL_START_EPOCH=$(date '+%s')
if $PY -m experiments.cuerator.final_analysis \
    --session "$SESSION_DIR" 2>&1; then
    echo "  Final analysis succeeded."
else
    echo "  Final analysis FAILED (exit $?)."
fi
log_end_and_elapsed "Final analysis" "$FINAL_START_EPOCH"
log_end_and_elapsed "Run" "$RUN_START_EPOCH"
echo "=== Run complete ==="
