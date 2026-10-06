#!/bin/bash
# Wrap a codex invocation in bwrap with role-specific filesystem isolation.
#
# Roles:
#   plan      — sees formulations/, results/, policy_results/, history.jsonl,
#               strategy_memo/ (RW), eval_protocol/ (RW), policy_protocol/ (RW),
#               oracle_reports/ (RO), policy_reports/ (RO). last_failure.txt
#               as RO if PLAN_LAST_FAILURE is set.
#   design    — sees formulations/ as RO; current strategy memo as RO single
#               file (mounted at /work/strategy_memo.md); writes the new fNNN.py
#               into /work/output (RW). Does NOT see history, results, or any
#               protocol/report directories.
#   oracle    — sees this iter's results.json, formulation.py, strategy_memo.md,
#               eval_protocol.json, raw.npz as RO single files; writes the
#               report into /work/oracle_report.md (single-file RW).
#   policy    — sees this iter's results.json, raw.npz, policy_results.json,
#               policy_raw.npz, formulation.py, strategy_memo.md,
#               eval_protocol.json, policy_protocol.json, oracle_report.md
#               as RO; writes /work/policy_report.md (single-file RW).
#
# Usage:
#   wrap_agent.sh <role> <session_dir> <conda_env_bin> -- codex_args...
#
# Env vars (per role):
#   plan      — PLAN_LAST_FAILURE     (optional path; RO bound at /work/last_failure.txt)
#   design    — DESIGN_CURRENT_MEMO   (required path; RO bound at /work/strategy_memo.md)
#               DESIGN_OUTPUT_DIR     (required dir;  RW bound at /work/output)
#   oracle    — ORACLE_RESULT_FILE      (required path; RO bound at /work/results.json)
#               ORACLE_FORMULATION_FILE (required path; RO bound at /work/formulation.py)
#               ORACLE_MEMO_FILE        (required path; RO bound at /work/strategy_memo.md)
#               EVAL_PROTOCOL_FILE    (required path; RO bound at /work/eval_protocol.json)
#               ORACLE_RAW_FILE         (required path; RO bound at /work/raw.npz)
#               ORACLE_REPORT_FILE      (required path; RW bound at /work/oracle_report.md)
#   policy    — POL_RESULT_FILE       (required path; RO bound at /work/results.json)
#               POL_RAW_FILE          (required path; RO bound at /work/raw.npz)
#               POL_POLICY_RESULT     (required path; RO bound at /work/policy_results.json)
#               POL_POLICY_RAW        (required path; RO bound at /work/policy_raw.npz)
#               POL_FORMULATION_FILE  (required path; RO bound at /work/formulation.py)
#               POL_MEMO_FILE         (required path; RO bound at /work/strategy_memo.md)
#               POL_EVAL_PROTOCOL     (required path; RO bound at /work/eval_protocol.json)
#               POL_POLICY_PROTOCOL   (required path; RO bound at /work/policy_protocol.json)
#               POL_ORACLE_REPORT  (required path; RO bound at /work/oracle_report.md)
#               POL_REPORT_FILE       (required path; RW bound at /work/policy_report.md)
#
# Inside the sandbox:
#   role=plan      → /work/{formulations,results,policy_results,history.jsonl,
#                          strategy_memo,eval_protocol,policy_protocol,
#                          oracle_reports,policy_reports,last_failure.txt?}
#   role=design    → /work/{formulations,strategy_memo.md,output}
#   role=oracle → /work/{results.json,formulation.py,strategy_memo.md,
#                          eval_protocol.json,raw.npz,oracle_report.md}
#   role=policy    → /work/{results.json,raw.npz,policy_results.json,
#                          policy_raw.npz,formulation.py,strategy_memo.md,
#                          eval_protocol.json,policy_protocol.json,
#                          oracle_report.md,policy_report.md}
#
# Everything else on the host is invisible.
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <role> <session_dir> <conda_env_bin> -- codex_args..." >&2
    exit 1
fi

ROLE="$1"
SESSION_DIR="$(cd "$2" && pwd)"
CONDA_ENV_BIN="$3"
shift 3
if [[ "${1:-}" == "--" ]]; then
    shift
fi

AGENT_CTX="$SESSION_DIR/agent_context"
mkdir -p "$AGENT_CTX/formulations" "$AGENT_CTX/results" "$AGENT_CTX/strategy_memo" \
         "$AGENT_CTX/eval_protocol" "$AGENT_CTX/policy_protocol" \
         "$AGENT_CTX/oracle_reports" "$AGENT_CTX/policy_reports" \
         "$AGENT_CTX/policy_results"
[[ -f "$AGENT_CTX/history.jsonl" ]] || touch "$AGENT_CTX/history.jsonl"

NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
CODEX_BIN_DIR="$(dirname "$(command -v codex)")"
CONDA_INSTALL="$(dirname "$(dirname "$(dirname "$CONDA_ENV_BIN")")")"

COMMON_BINDS=(
    --ro-bind /usr /usr
    --ro-bind /lib /lib
    --ro-bind /lib64 /lib64
    --ro-bind /etc /etc
    --ro-bind /bin /bin
    --ro-bind /sbin /sbin
    --ro-bind "$CONDA_INSTALL" "$CONDA_INSTALL"
    --bind "${CODEX_HOME:-$HOME/.codex}" "${CODEX_HOME:-$HOME/.codex}"
    --proc /proc
    --dev /dev
    --tmpfs /tmp
    --setenv HOME "$HOME"
    --setenv CODEX_HOME "${CODEX_HOME:-$HOME/.codex}"
    --setenv PATH "$CONDA_ENV_BIN:$CODEX_BIN_DIR:/usr/bin:/bin"
    --setenv USER "${USER:-$(id -un)}"
    --share-net
    --chdir /work
)

# Optional host paths: bwrap aborts on a --ro-bind whose source is absent, so these
# are only added when they actually exist (systemd-resolved, nvm-installed node).
for opt in /run/systemd/resolve "$NVM_DIR"; do
    [[ -e "$opt" ]] && COMMON_BINDS+=(--ro-bind "$opt" "$opt")
done

# The codex launcher may live outside the standard prefixes (e.g. a local
# install directory). Bind its real location read-only when that is the case.
# `npm install -g` puts the launcher under <prefix>/lib/node_modules and the
# native binary in a sibling package there, so bind that root too - the bin
# entry is only a symlink into it. Anchor on node_modules rather than a fixed
# depth so a non-npm install cannot walk up to /.
CODEX_REAL_DIR="$(cd "$CODEX_BIN_DIR" && pwd -P)"
CODEX_REAL="$(readlink -f "$(command -v codex)")"
case "$CODEX_REAL" in
    */node_modules/*) CODEX_PKG_ROOT="${CODEX_REAL%%/node_modules/*}/node_modules" ;;
    *) CODEX_PKG_ROOT="$(dirname "$CODEX_REAL")" ;;
esac
for codex_dir in "$CODEX_REAL_DIR" "$CODEX_PKG_ROOT"; do
    case "$codex_dir" in
        /usr/*|/bin/*|/sbin/*|"$CONDA_INSTALL"/*|"$NVM_DIR"/*) ;;
        *) COMMON_BINDS+=(--ro-bind "$codex_dir" "$codex_dir") ;;
    esac
done

ROLE_BINDS=(--tmpfs /work)

case "$ROLE" in
    plan)
        ROLE_BINDS+=(
            --ro-bind "$AGENT_CTX/formulations" /work/formulations
            --ro-bind "$AGENT_CTX/results" /work/results
            --ro-bind "$AGENT_CTX/policy_results" /work/policy_results
            --ro-bind "$AGENT_CTX/history.jsonl" /work/history.jsonl
            --ro-bind "$AGENT_CTX/oracle_reports" /work/oracle_reports
            --ro-bind "$AGENT_CTX/policy_reports" /work/policy_reports
            --bind "$AGENT_CTX/strategy_memo" /work/strategy_memo
            --bind "$AGENT_CTX/eval_protocol" /work/eval_protocol
            --bind "$AGENT_CTX/policy_protocol" /work/policy_protocol
        )
        if [[ -n "${PLAN_LAST_FAILURE:-}" ]] && [[ -f "$PLAN_LAST_FAILURE" ]]; then
            ROLE_BINDS+=(--ro-bind "$PLAN_LAST_FAILURE" /work/last_failure.txt)
        fi
        ;;
    design)
        if [[ -z "${DESIGN_CURRENT_MEMO:-}" ]] || [[ ! -f "$DESIGN_CURRENT_MEMO" ]]; then
            echo "wrap_agent.sh design: DESIGN_CURRENT_MEMO must be a readable file" >&2
            exit 1
        fi
        if [[ -z "${DESIGN_OUTPUT_DIR:-}" ]]; then
            echo "wrap_agent.sh design: DESIGN_OUTPUT_DIR must be set" >&2
            exit 1
        fi
        mkdir -p "$DESIGN_OUTPUT_DIR"
        ROLE_BINDS+=(
            --ro-bind "$AGENT_CTX/formulations" /work/formulations
            --ro-bind "$DESIGN_CURRENT_MEMO" /work/strategy_memo.md
            --bind "$DESIGN_OUTPUT_DIR" /work/output
        )
        ;;
    oracle)
        for var in ORACLE_RESULT_FILE ORACLE_FORMULATION_FILE ORACLE_MEMO_FILE EVAL_PROTOCOL_FILE ORACLE_RAW_FILE ORACLE_REPORT_FILE; do
            val="${!var:-}"
            if [[ -z "$val" ]]; then
                echo "wrap_agent.sh oracle: $var must be set" >&2
                exit 1
            fi
        done
        for var in ORACLE_RESULT_FILE ORACLE_FORMULATION_FILE ORACLE_MEMO_FILE EVAL_PROTOCOL_FILE ORACLE_RAW_FILE; do
            val="${!var}"
            if [[ ! -f "$val" ]]; then
                echo "wrap_agent.sh oracle: $var ($val) is not a readable file" >&2
                exit 1
            fi
        done
        : > "$ORACLE_REPORT_FILE"
        ROLE_BINDS+=(
            --ro-bind "$ORACLE_RESULT_FILE" /work/results.json
            --ro-bind "$ORACLE_FORMULATION_FILE" /work/formulation.py
            --ro-bind "$ORACLE_MEMO_FILE" /work/strategy_memo.md
            --ro-bind "$EVAL_PROTOCOL_FILE" /work/eval_protocol.json
            --ro-bind "$ORACLE_RAW_FILE" /work/raw.npz
            --bind "$ORACLE_REPORT_FILE" /work/oracle_report.md
        )
        ;;
    policy)
        for var in POL_RESULT_FILE POL_RAW_FILE POL_POLICY_RESULT POL_POLICY_RAW \
                   POL_FORMULATION_FILE POL_MEMO_FILE POL_EVAL_PROTOCOL POL_POLICY_PROTOCOL \
                   POL_ORACLE_REPORT POL_REPORT_FILE; do
            val="${!var:-}"
            if [[ -z "$val" ]]; then
                echo "wrap_agent.sh policy: $var must be set" >&2
                exit 1
            fi
        done
        for var in POL_RESULT_FILE POL_RAW_FILE POL_POLICY_RESULT POL_POLICY_RAW \
                   POL_FORMULATION_FILE POL_MEMO_FILE POL_EVAL_PROTOCOL POL_POLICY_PROTOCOL \
                   POL_ORACLE_REPORT; do
            val="${!var}"
            if [[ ! -f "$val" ]]; then
                echo "wrap_agent.sh policy: $var ($val) is not a readable file" >&2
                exit 1
            fi
        done
        : > "$POL_REPORT_FILE"
        ROLE_BINDS+=(
            --ro-bind "$POL_RESULT_FILE" /work/results.json
            --ro-bind "$POL_RAW_FILE" /work/raw.npz
            --ro-bind "$POL_POLICY_RESULT" /work/policy_results.json
            --ro-bind "$POL_POLICY_RAW" /work/policy_raw.npz
            --ro-bind "$POL_FORMULATION_FILE" /work/formulation.py
            --ro-bind "$POL_MEMO_FILE" /work/strategy_memo.md
            --ro-bind "$POL_EVAL_PROTOCOL" /work/eval_protocol.json
            --ro-bind "$POL_POLICY_PROTOCOL" /work/policy_protocol.json
            --ro-bind "$POL_ORACLE_REPORT" /work/oracle_report.md
            --bind "$POL_REPORT_FILE" /work/policy_report.md
        )
        ;;
    *)
        echo "wrap_agent.sh: unknown role '$ROLE' (expected: plan, design, oracle, policy)" >&2
        exit 1
        ;;
esac

exec bwrap "${COMMON_BINDS[@]}" "${ROLE_BINDS[@]}" -- "$@"
