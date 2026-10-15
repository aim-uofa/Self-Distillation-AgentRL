#!/usr/bin/env bash
# Shared Conda discovery; safe to source without creating/activating an env.

agentrl_init_conda() {
    if [[ -z "${CONDA_BASE:-}" ]]; then
        if command -v conda >/dev/null 2>&1; then
            CONDA_BASE="$(conda info --base)" || return
        elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
            # Activated shells export CONDA_EXE even when the shell function
            # and condabin are not inherited by a child Bash process.
            CONDA_BASE="$("${CONDA_EXE}" info --base)" || return
        else
            echo "Conda was not found. Put conda on PATH or set CONDA_BASE to its installation directory." >&2
            return 1
        fi
    fi
    if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        echo "Conda initialization script not found under ${CONDA_BASE}" >&2
        return 1
    fi
    CONDA_BASE="$(cd -- "${CONDA_BASE}" && pwd)" || return
    export CONDA_BASE
    local restore_nounset=0 status=0
    [[ "$-" == *u* ]] && restore_nounset=1
    set +u
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh" || status=$?
    if (( restore_nounset )); then set -u; fi
    return "${status}"
}

agentrl_activate_conda() {
    agentrl_init_conda || return
    export CONDA_ENV="${CONDA_ENV:-agentrl}"
    local restore_nounset=0 status=0
    [[ "$-" == *u* ]] && restore_nounset=1
    set +u
    conda activate "${CONDA_ENV}" || status=$?
    if (( restore_nounset )); then set -u; fi
    if (( status )); then return "${status}"; fi
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    export JAVA_HOME="${CONDA_PREFIX}"
    hash -r
}
