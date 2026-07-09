#!/usr/bin/env bash
set -euo pipefail

# Prepare external depth-estimator dependencies used by run_depth_ablation_t01.sh.

WORKDIR="${WORKDIR:-/mnt/data/wangqq}"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/DreamScene360}"

DREAM_ENV="${DREAM_ENV:-${WORKDIR}/conda_envs/dreamscene360}"
DA3_ENV="${DA3_ENV:-${WORKDIR}/conda_envs/depth_anything3}"
DAP_ENV="${DAP_ENV:-${WORKDIR}/conda_envs/dap}"

DA3_ROOT="${DA3_ROOT:-${WORKDIR}/depth-anything-3}"
VGGT_ROOT="${VGGT_ROOT:-${WORKDIR}/vggt}"
DAP_ROOT="${DAP_ROOT:-${WORKDIR}/DAP}"
DAP_WEIGHTS_DIR="${DAP_WEIGHTS_DIR:-${WORKDIR}/DAP-weights}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-${WORKDIR}/hf_cache}"
TORCH_HOME="${TORCH_HOME:-${WORKDIR}/torch_cache}"

INSTALL_DA3="${INSTALL_DA3:-1}"
INSTALL_VGGT="${INSTALL_VGGT:-1}"
INSTALL_DAP="${INSTALL_DAP:-1}"

source_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return
  fi

  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "${WORKDIR}/miniconda3/etc/profile.d/conda.sh"
    "${WORKDIR}/anaconda3/etc/profile.d/conda.sh"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck source=/dev/null
      source "$candidate"
      return
    fi
  done

  echo "[error] conda not found. Activate conda first or set up conda.sh." >&2
  exit 1
}

conda_env_exists() {
  [[ -x "$1/bin/python" ]]
}

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [[ -d "$dir/.git" ]]; then
    echo "[repo] exists: $dir"
  elif [[ -d "$dir" ]]; then
    echo "[repo] directory exists but is not a git repo: $dir"
  else
    git clone "$url" "$dir"
  fi
}

export HF_ENDPOINT HF_HOME TORCH_HOME
mkdir -p "$HF_HOME" "$TORCH_HOME"

source_conda

if [[ "$INSTALL_DA3" == "1" ]]; then
  echo "[setup] Depth Anything 3"
  clone_or_update "https://github.com/ByteDance-Seed/depth-anything-3.git" "$DA3_ROOT"
  if ! conda_env_exists "$DA3_ENV"; then
    conda create -p "$DA3_ENV" python=3.10 -y
  fi
  conda activate "$DA3_ENV"
  python -m pip install --upgrade pip
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  cd "$DA3_ROOT"
  python -m pip install -e .
fi

if [[ "$INSTALL_VGGT" == "1" ]]; then
  echo "[setup] VGGT"
  clone_or_update "https://github.com/facebookresearch/vggt.git" "$VGGT_ROOT"
  echo "[setup] VGGT is used from source via VGGT_ROOT=${VGGT_ROOT}."
  echo "[setup] Not installing VGGT requirements into DreamScene360 because they"
  echo "[setup] can downgrade torch/numpy and break the Python 3.8 environment."
fi

if [[ "$INSTALL_DAP" == "1" ]]; then
  echo "[setup] DAP"
  clone_or_update "https://github.com/Insta360-Research-Team/DAP.git" "$DAP_ROOT"
  if ! conda_env_exists "$DAP_ENV"; then
    conda create -p "$DAP_ENV" python=3.12 -y
  fi
  conda activate "$DAP_ENV"
  python -m pip install --upgrade pip
  python -m pip install torch==2.7.1 torchvision==0.22.1
  cd "$DAP_ROOT"
  python -m pip install -r requirements.txt
  python -m pip install huggingface_hub pyyaml
  if [[ ! -d "$DAP_WEIGHTS_DIR" || -z "$(find "$DAP_WEIGHTS_DIR" -maxdepth 1 -type f 2>/dev/null | head -1)" ]]; then
    huggingface-cli download Insta360-Research/DAP-weights --local-dir "$DAP_WEIGHTS_DIR"
  else
    echo "[setup] DAP weights already exist: $DAP_WEIGHTS_DIR"
  fi
fi

echo "[done] depth ablation environments are prepared"
echo "[done] next:"
echo "  cd ${PROJECT_DIR}"
echo "  AUTO_SKIP_UNAVAILABLE=0 bash scripts/run_depth_ablation_t01.sh"
