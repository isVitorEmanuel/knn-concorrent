#!/usr/bin/env bash
# =============================================================================
# run_remaining_final.sh  —  Execuções específicas restantes
# =============================================================================

set -euo pipefail

LOOPS="${1:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_knn_test.sh"

[[ ! -f "${RUN_SCRIPT}" ]] && echo "❌ run_knn_test.sh não encontrado em ${SCRIPT_DIR}" && exit 1
[[ ! -x "${RUN_SCRIPT}" ]] && chmod +x "${RUN_SCRIPT}"

# Definição dos testes específicos: Versão, GC, Threads
TESTES=(
  "Virtual,ZGC,50"
  "Virtual,G1GC,50"
  "Virtual,G1GC,100"
  "Platform,ParallelGC,100"
)

TOTAL=${#TESTES[@]}
ATUAL=0
FALHAS=()

echo "════════════════════════════════════════════════════════"
echo "  🚀 Executando testes remanescentes específicos"
echo "════════════════════════════════════════════════════════"
echo "  Total: ${TOTAL} execuções"
echo "════════════════════════════════════════════════════════"
echo ""

START_TOTAL=$(date +%s)

run_test() {
  local VERSAO=$1 GC=$2 T=$3
  ATUAL=$((ATUAL + 1))
  echo "────────────────────────────────────────────────────────"
  echo "  [${ATUAL}/${TOTAL}] ${VERSAO} | ${GC} | ${T} threads | ${LOOPS} loops"
  echo "────────────────────────────────────────────────────────"
  local START=$(date +%s)
  if "${RUN_SCRIPT}" "${VERSAO}" "${GC}" "${T}" "${LOOPS}"; then
    local END=$(date +%s)
    echo "  ✅ Concluído em $(( (END-START)/60 ))m $(( (END-START)%60 ))s"
  else
    local END=$(date +%s)
    echo "  ❌ FALHOU após $(( (END-START)/60 ))m $(( (END-START)%60 ))s"
    FALHAS+=("${VERSAO}_${GC}_${T}t")
  fi
  echo ""
}

# Executa cada item da lista
for item in "${TESTES[@]}"; do
  IFS=',' read -r v gc t <<< "$item"
  run_test "$v" "$gc" "$t"
done

# ─── Resumo ──────────────────────────────────────────────────────────────────
END_TOTAL=$(date +%s)
ELAPSED_TOTAL=$((END_TOTAL - START_TOTAL))

echo "════════════════════════════════════════════════════════"
echo "  ✅ Batch finalizado"
echo "════════════════════════════════════════════════════════"
echo "  Tempo total : $((ELAPSED_TOTAL/3600))h $(( (ELAPSED_TOTAL%3600)/60 ))m $((ELAPSED_TOTAL%60))s"
if [[ ${#FALHAS[@]} -gt 0 ]]; then
  echo "  ⚠️  Falhas:"
  for F in "${FALHAS[@]}"; do echo "     - ${F}"; done
else
  echo "  🎉 Todas as execuções completaram sem erros"
fi
echo "════════════════════════════════════════════════════════"
