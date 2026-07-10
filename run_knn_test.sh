set -euo pipefail

VERSAO="${1:-Serial}"
GC="${2:-G1GC}"
THREADS="${3:-20}"
LOOPS="${4:-3}"

case "${GC^^}" in
  PARALLELGC|PARALLEL) GC_FLAG="-XX:-UseG1GC -XX:+UseParallelGC";              GC_LABEL="ParallelGC"; GC_EXTRA="" ;;
  G1GC|G1)             GC_FLAG="-XX:+UseG1GC";                                 GC_LABEL="G1GC";       GC_EXTRA="-XX:MaxGCPauseMillis=100 -XX:G1ReservePercent=20" ;;
  ZGC|Z)               GC_FLAG="-XX:-UseG1GC -XX:+UseZGC -XX:+ZGenerational";  GC_LABEL="ZGC";        GC_EXTRA="" ;;
  *) echo "❌ GC inválido: ${GC}. Use: ParallelGC | G1GC | ZGC"; exit 1 ;;
esac

VERSOES_VALIDAS=("Serial" "Platform" "Virtual" "Hibrid" "Sync" "Lock" "Semaphore" "Barrier" "Atomic" "ForkJoin" "StructuredConcurrency" "Executor" "CallableFuture" "ParallelStream" "CompletableFuture")
VERSAO_VALIDA=false
for V in "${VERSOES_VALIDAS[@]}"; do
  [[ "${V,,}" == "${VERSAO,,}" ]] && VERSAO_VALIDA=true && VERSAO="${V}" && break
done
[[ "${VERSAO_VALIDA}" == false ]] && echo "❌ Versão inválida: ${VERSAO}. Disponíveis: ${VERSOES_VALIDAS[*]}" && exit 1

case "${VERSAO}" in
  Serial)    CLASSE="KNNSerial";    PACOTE="serial"    ;;
  Platform)  CLASSE="KNNPlatform";  PACOTE="platform"  ;;
  Virtual)   CLASSE="KNNVirtual";   PACOTE="virtual"   ;;
  Hibrid)    CLASSE="KNNHibrid";    PACOTE="hibrid"    ;;
  Sync)      CLASSE="KNNSync";      PACOTE="sync"      ;;
  Lock)      CLASSE="KNNLock";      PACOTE="lock"      ;;
  Semaphore) CLASSE="KNNSemaphore"; PACOTE="semaphore" ;;
  Barrier)   CLASSE="KNNBarrier";   PACOTE="barrier"   ;;
  Atomic)    CLASSE="KNNAtomic";    PACOTE="atomic"    ;;
  ForkJoin)              CLASSE="KNNForkJoin";              PACOTE="executor" ;;
  StructuredConcurrency) CLASSE="KNNStructuredConcurrency"; PACOTE="executor" ;;
  Executor)              CLASSE="KNNExecutor";              PACOTE="executor" ;;
  CallableFuture)        CLASSE="KNNCallableFuture";        PACOTE="executor" ;;
  ParallelStream)        CLASSE="KNNParallelStream";        PACOTE="executor" ;;
  CompletableFuture)     CLASSE="KNNCompletableFuture";     PACOTE="executor" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAR_PATH="${SCRIPT_DIR}/target/benchmarks.jar"
RESULTS_DIR="${SCRIPT_DIR}/results/${VERSAO}_${GC_LABEL}_${THREADS}t_$(date +%Y%m%d_%H%M%S)"
JFR_FILE="${RESULTS_DIR}/recording.jfr"
JMX_FILE="${RESULTS_DIR}/knn_test.jmx"
GROOVY_FILE="${RESULTS_DIR}/knn_sampler.groovy"
JMETER_RESULTS="${RESULTS_DIR}/jmeter_results.jtl"
JMETER_REPORT="${RESULTS_DIR}/jmeter_report"

mkdir -p "${RESULTS_DIR}"

echo "Verificando dependências..."
command -v java   >/dev/null 2>&1 || { echo "❌ java não encontrado";   exit 1; }
command -v mvn    >/dev/null 2>&1 || { echo "❌ maven não encontrado";  exit 1; }
command -v jmeter >/dev/null 2>&1 || { echo "❌ jmeter não encontrado"; exit 1; }

JMETER_HOME="$(realpath "$(dirname "$(command -v jmeter)")/..")"
echo "   JMeter home: ${JMETER_HOME}"

if [[ ! -f "${JAR_PATH}" ]]; then
  echo "🔨 Buildando com Maven..."
  cd "${SCRIPT_DIR}" && mvn clean package -DskipTests -q
fi

JAR_DEST="${JMETER_HOME}/lib/ext/knn-concorrent.jar"
if [[ ! -f "${JAR_DEST}" ]] || [[ "${JAR_PATH}" -nt "${JAR_DEST}" ]]; then
  echo "📦 Copiando JAR para ${JMETER_HOME}/lib/ext/..."
  cp "${JAR_PATH}" "${JAR_DEST}"
  echo "   ✅ JAR copiado"
else
  echo "   ✅ JAR já está em lib/ext"
fi

echo "📝 Gerando script Groovy: ${GROOVY_FILE}"
cat > "${GROOVY_FILE}" << GROOVY
import ${PACOTE}.${CLASSE}
import generator.DataSetGenerator
import classes.Neighbor
import java.util.ArrayList

try {
    def k = 21
    def filePath = "${SCRIPT_DIR}/dataset_high_dim.csv"
    int numFeatures = DataSetGenerator.NUM_FEATURES
    def targetValues = new ArrayList<Double>()
    for (int i = 0; i < numFeatures; i++) { targetValues.add(500.0d) }
    def target = new Neighbor(targetValues, "Unknown")

    def knn = new ${CLASSE}()
    def result = knn.predictStream(filePath, target, k)
    SampleResult.setResponseData(result.toString(), "UTF-8")
    SampleResult.setSuccessful(true)
} catch (Throwable t) {
    SampleResult.setSuccessful(false)
    SampleResult.setResponseData("CORREÇÃO LOG - Erro na execução: " + t.toString(), "UTF-8")
    log.error("Falha catastrófica no sampler Concorrente do Groovy", t)
}
GROOVY
echo "   ✅ Groovy gerado"

echo "📄 Gerando plano JMeter: ${JMX_FILE}"
cat > "${JMX_FILE}" << JMXEOF
<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0" jmeter="5.6">
  <hashTree>
    <TestPlan guiclass="TestPlanGui" testclass="TestPlan"
              testname="KNN ${VERSAO} - ${GC_LABEL} - ${THREADS} threads x ${LOOPS} loops" enabled="true">
      <boolProp name="TestPlan.functional_mode">false</boolProp>
      <boolProp name="TestPlan.serialize_threadgroups">false</boolProp>
    </TestPlan>
    <hashTree>

      <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup"
                   testname="${THREADS} usuários x ${LOOPS} loops" enabled="true">
        <intProp name="ThreadGroup.num_threads">${THREADS}</intProp>
        <intProp name="ThreadGroup.ramp_time">0</intProp>
        <boolProp name="ThreadGroup.scheduler">false</boolProp>
        <elementProp name="ThreadGroup.main_controller" elementType="LoopController">
          <boolProp name="LoopController.continue_forever">false</boolProp>
          <intProp name="LoopController.loops">${LOOPS}</intProp>
        </elementProp>
      </ThreadGroup>
      <hashTree>
        <JSR223Sampler guiclass="TestBeanGUI" testclass="JSR223Sampler"
                       testname="KNN ${VERSAO}" enabled="true">
          <stringProp name="scriptLanguage">groovy</stringProp>
          <stringProp name="filename">${GROOVY_FILE}</stringProp>
          <stringProp name="script"></stringProp>
          <stringProp name="cacheKey">true</stringProp>
        </JSR223Sampler>
        <hashTree/>
        <ResultCollector guiclass="SimpleDataWriter" testclass="ResultCollector"
                         testname="Save Results" enabled="true">
          <boolProp name="ResultCollector.error_logging">false</boolProp>
          <objProp>
            <name>saveConfig</name>
            <value class="SampleSaveConfiguration">
              <time>true</time><latency>true</latency><timestamp>true</timestamp>
              <success>true</success><label>true</label><code>true</code>
              <message>true</message><threadName>true</threadName>
              <bytes>true</bytes><sentBytes>true</sentBytes>
              <threadCounts>true</threadCounts><idleTime>true</idleTime>
              <connectTime>true</connectTime><fieldNames>true</fieldNames>
            </value>
          </objProp>
          <stringProp name="filename">${JMETER_RESULTS}</stringProp>
        </ResultCollector>
        <hashTree/>
      </hashTree>

    </hashTree>
  </hashTree>
</jmeterTestPlan>
JMXEOF
echo "   ✅ JMX gerado (${THREADS} threads × ${LOOPS} loops = $((THREADS * LOOPS)) execuções)"

echo ""
echo "🚀 Iniciando JMeter + JFR..."
echo "   Versão  : ${VERSAO} (${CLASSE})"
echo "   GC      : ${GC_LABEL} (${GC_FLAG})"
echo "   Threads : ${THREADS}"
echo "   Loops   : ${LOOPS} por thread"
echo "   Total   : $((THREADS * LOOPS)) execuções"
echo "   JFR     : ${JFR_FILE}"
echo ""

JFR_TMP_DIR="$(mktemp -d /tmp/knn_jfr.XXXXXX)"
JFR_TMP_FILE="${JFR_TMP_DIR}/recording.jfr"

export JVM_ARGS="${GC_FLAG} ${GC_EXTRA} --enable-preview -XX:+FlightRecorder -XX:StartFlightRecording=filename=${JFR_TMP_FILE},dumponexit=true,settings=profile,name=knn-${VERSAO}"
export HEAP="-Xms512m -Xmx2g"

jmeter -n \
  -t "${JMX_FILE}" \
  -l "${JMETER_RESULTS}" \
  -e -o "${JMETER_REPORT}" \
  -Jjmeter.save.saveservice.assertion_results_failure_message=true \
  -Jjmeter.save.saveservice.data_type=true

if [[ -f "${JFR_TMP_FILE}" ]]; then
  mv "${JFR_TMP_FILE}" "${JFR_FILE}"
  rmdir "${JFR_TMP_DIR}" 2>/dev/null || true
else
  echo "⚠️  Aviso: arquivo JFR temporário não encontrado em ${JFR_TMP_FILE}"
fi

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ Teste concluído"
echo "════════════════════════════════════════════"
echo "  Versão     : ${VERSAO}"
echo "  GC         : ${GC_LABEL}"
echo "  Threads    : ${THREADS}"
echo "  Loops      : ${LOOPS} por thread"
echo "  Total      : $((THREADS * LOOPS)) execuções"