const $ = (id) => document.getElementById(id);
const state = { chart: null, candleSeries: null, lastSelection: null, busy: false, observer: null, selectionVersion: 0 };

const fmt = {
  price: (v) => (v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v))) ? Number(v).toLocaleString("ko-KR", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—",
  pct: (v, digits = 2) => (v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v))) ? `${(Number(v) * 100).toFixed(digits)}%` : "—",
  signedPct: (v, digits = 2) => (v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v))) ? `${Number(v) >= 0 ? "+" : ""}${(Number(v) * 100).toFixed(digits)}%` : "—",
  num: (v, digits = 4) => (v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v))) ? Number(v).toFixed(digits) : "—",
  int: (v) => (v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v))) ? Number(v).toLocaleString("ko-KR") : "—",
  time: (v) => v ? new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short", timeZone: "Asia/Seoul" }).format(new Date(v)) : "—",
};

function selection() {
  const token = $("token").value.trim().toUpperCase();
  const timeframe = $("timeframe").value;
  if (!/^[A-Z0-9]{2,30}$/.test(token)) throw new Error("코인 이름은 영문 대문자와 숫자로 입력하세요. 예: BTC");
  return { token, timeframe };
}

function updateBadges() {
  try {
    const { token, timeframe } = selection();
    $("badgeToken").textContent = `${token}USDT`;
    $("badgeTimeframe").textContent = timeframe.toUpperCase();
  } catch (_) {}
}

function setBusy(busy, title = "", message = "") {
  state.busy = busy;
  $("token").disabled = busy;
  $("timeframe").disabled = busy;
  document.querySelectorAll("[data-action]").forEach((button) => button.disabled = busy);
  $("statusPanel").classList.toggle("running", busy);
  $("statusPanel").classList.remove("failed");
  if (title) $("statusTitle").textContent = title;
  if (message) $("statusMessage").textContent = message;
}

function fail(message) {
  setBusy(false, "요청을 완료하지 못했습니다", message);
  $("statusPanel").classList.add("failed");
  toast(message);
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), 4200);
}

async function requestJson(url, options = {}) {
  if (location.protocol === "file:") throw new Error("HTML 파일을 직접 열면 연결되지 않습니다. 서버를 실행한 뒤 http://localhost:8000 으로 접속하세요.");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = Array.isArray(data?.detail) ? data.detail.map(e => e.msg).join(" · ") : data?.detail;
      const error = new Error(detail || `서버 응답 오류 (${response.status})`);
      error.status = response.status;
      throw error;
    }
    if (!data || typeof data !== "object") throw new Error("서버가 JSON 대신 다른 페이지를 반환했습니다. Python 서버 주소를 확인하세요.");
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("서버 응답이 지연되고 있습니다. 연결 상태를 확인해 주세요.");
    if (error instanceof TypeError) throw new Error("서버에 연결할 수 없습니다. Python 서버 실행 상태와 접속 주소를 확인하세요.");
    throw error;
  } finally { clearTimeout(timer); }
}

async function startAction(action) {
  if (state.busy) return;
  let current;
  try { current = selection(); } catch (error) { return fail(error.message); }
  if (action === "lockbox" && !$("lockboxConfirm").checked) {
    return fail("최종 미공개 데이터 검증 확인란을 먼저 체크하세요.");
  }
  updateBadges();
  const labels = {
    predict: ["최신 예측을 준비하고 있습니다", "처음 실행하거나 모델을 다시 학습하면 몇 분 걸릴 수 있습니다."],
    refresh: ["종료된 캔들을 확인하고 있습니다", "Binance 공개 데이터에서 새 캔들만 가져옵니다."],
    validate: ["시간순 모의시험을 실행하고 있습니다", "여러 구간을 차례로 학습하므로 가장 오래 걸리는 작업입니다."],
    lockbox: ["최종 미공개 데이터를 검증하고 있습니다", "연구가 끝난 뒤 한 번만 보는 마지막 시험입니다."],
  };
  setBusy(true, ...labels[action]);
  try {
    const started = await requestJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...current, noise_test: $("noiseTest").checked, confirm_lockbox: $("lockboxConfirm").checked }),
    });
    sessionStorage.setItem("activeJob", JSON.stringify({ id: started.job_id, ...current }));
    await pollJob(started.job_id);
  } catch (error) { fail(error.message); }
}

async function pollJob(jobId) {
  let failures = 0;
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    let job;
    try { job = await requestJson(`/api/jobs/${jobId}`); failures = 0; }
    catch (error) {
      if (error.status === 404) { sessionStorage.removeItem("activeJob"); throw new Error("서버가 재시작되었거나 작업이 만료되었습니다. 다시 실행하세요."); }
      if (++failures >= 5) throw new Error("작업 상태 연결이 끊겼습니다. 페이지를 새로고침하면 저장된 작업에 다시 연결합니다.");
      $("statusMessage").textContent = `서버 재연결 중 (${failures}/5)…`;
      await new Promise(resolve => setTimeout(resolve, failures * 2000));
      continue;
    }
    $("operationLog").textContent = job.log || "처리 중…";
    $("statusMessage").textContent = job.message || "계산 중입니다.";
    if (job.status === "failed") {
      sessionStorage.removeItem("activeJob");
      $("operationLog").textContent = job.log || `${job.error_type || "Error"}: ${job.message}`;
      return fail(job.message);
    }
    if (job.status === "complete") {
      sessionStorage.removeItem("activeJob");
      $("operationLog").textContent = job.log || "작업은 정상 완료되었으며 별도 출력은 없습니다.";
      renderPayload(job.result || {});
      await loadChart();
      setBusy(false, "완료되었습니다", [job.message, ...(job.result?.warnings || [])].join(" · "));
      toast(job.message);
      return;
    }
  }
}

function renderPayload(payload) {
  if (payload.prediction) renderPrediction(payload.prediction);
  if (payload.range_prediction) renderRange(payload.range_prediction);
  else if (payload.prediction) renderRange({});
  if (payload.lockbox_result) $("operationLog").textContent += "\n최종 검증 결과\n" + JSON.stringify(payload.lockbox_result, null, 2);
  if (payload.validation_summary) renderValidation(payload.validation_summary);
}

function setSignal(signal) {
  const clean = String(signal || "WAITING").toUpperCase();
  const badge = $("signalBadge");
  badge.textContent = clean;
  badge.className = `signal ${clean === "LONG" ? "long" : clean === "SHORT" ? "short" : "neutral"}`;
  $("gaugeSignal").textContent = clean === "WAITING" ? "대기 중" : clean;
}

function renderPrediction(result) {
  $("referenceClose").textContent = fmt.price(result.reference_close);
  $("upProbability").textContent = fmt.pct(result.probability_up);
  $("downProbability").textContent = fmt.pct(result.probability_down);
  $("threshold").textContent = fmt.pct(result.threshold);
  $("upMini").textContent = fmt.pct(result.probability_up, 1);
  $("downMini").textContent = fmt.pct(result.probability_down, 1);
  $("upFill").style.width = `${Math.max(0, Math.min(100, result.probability_up * 100))}%`;
  setSignal(result.signal);

  const score = Math.max(0, Math.min(100, Number(result.probability_up) * 100));
  $("gaugeValue").textContent = `${score.toFixed(1)}%`;
  $("needle").style.transform = `rotate(${-90 + score * 1.8}deg)`;
  $("predictionMeta").innerHTML = `기준 캔들 <b>${fmt.time(result.source_candle_start)} ~ ${fmt.time(result.source_candle_end)}</b><br>예측 대상 <b>${fmt.time(result.target_candle_start)} ~ ${fmt.time(result.target_candle_end)}</b>`;
  $("directionExplanation").textContent = `${result.signal} 신호입니다. 상승 ${fmt.pct(result.probability_up)}, 하락 ${fmt.pct(result.probability_down)}이며 신호 기준은 ${fmt.pct(result.threshold)}입니다.`;
}

function colorSigned(element, value) {
  element.classList.remove("positive", "negative");
  if (Number.isFinite(Number(value))) element.classList.add(Number(value) >= 0 ? "positive" : "negative");
}

function renderRange(result) {
  [["q10", result.q10_price, result.q10_return], ["q50", result.q50_price, result.q50_return], ["q90", result.q90_price, result.q90_return]].forEach(([q, price, change]) => {
    $(`${q}Price`).textContent = `${fmt.price(price)} USDT`;
    $(`${q}Return`).textContent = fmt.signedPct(change);
    colorSigned($(`${q}Return`), change);
  });
  const quality = result.quality_summary || {};
  $("coverage").textContent = fmt.pct(quality.interval_coverage);
  $("pinball").textContent = fmt.signedPct(quality.pinball_skill_score);
}

function renderValidation(summary) {
  $("rocAuc").textContent = fmt.num(summary.roc_auc);
  $("brier").textContent = fmt.num(summary.brier_score ?? summary.brier);
  $("gap").textContent = fmt.num(summary.generalization_gap, 3);
  $("foldStd").textContent = fmt.num(summary.fold_auc_std, 3);
  $("oosSamples").textContent = fmt.int(summary.oos_samples);
  $("foldCount").textContent = fmt.int(summary.folds);
  $("overfitStatus").textContent = summary.overfit_status || "검증 완료";
  const strategy = summary.strategy || {};
  $("trades").textContent = fmt.int(strategy.closed_trades);
  $("winRate").textContent = fmt.pct(strategy.win_rate);
  $("return").textContent = fmt.signedPct(strategy.cumulative_return);
  $("mdd").textContent = fmt.signedPct(strategy.mdd);
  $("profitFactor").textContent = fmt.num(strategy.profit_factor, 2);
  $("sharpe").textContent = fmt.num(strategy.sharpe, 2);
  colorSigned($("return"), strategy.cumulative_return);
  colorSigned($("mdd"), strategy.mdd);
}

function destroyChart() {
  state.observer?.disconnect();
  state.observer = null;
  if (state.chart) state.chart.remove();
  state.chart = null;
  $("chart").replaceChildren();
}

async function loadChart(limit = 2000) {
  const current = selection();
  const version = state.selectionVersion;
  try {
    const data = await requestJson(`/api/chart?token=${encodeURIComponent(current.token)}&timeframe=${encodeURIComponent(current.timeframe)}&limit=${limit}`);
    if (version !== state.selectionVersion) return;
    renderChart(data);
  } catch (error) {
    if (version !== state.selectionVersion) return;
    destroyChart();
    $("chartEmpty").style.display = "grid";
    $("chartEmpty").textContent = error.message;
  }
}

function renderChart(data) {
  destroyChart();
  if (!window.LightweightCharts) throw new Error("차트 라이브러리를 불러오지 못했습니다. 페이지를 새로고침하세요.");
  if (!data.candles?.length) throw new Error("표시할 종료 캔들이 없습니다.");
  $("chartEmpty").style.display = "none";
  const container = $("chart");
  const chart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight,
    layout: { background: { color: "#0d141b" }, textColor: "#8190a0", fontFamily: "IBM Plex Mono" },
    grid: { vertLines: { color: "rgba(129,148,165,.07)" }, horzLines: { color: "rgba(129,148,165,.10)" } },
    rightPriceScale: { borderColor: "#22303b", scaleMargins: { top: .08, bottom: .24 } },
    timeScale: { borderColor: "#22303b", timeVisible: true, secondsVisible: false, rightOffset: 8, barSpacing: 8, minBarSpacing: .5 },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
  });
  const candles = chart.addCandlestickSeries({ upColor: "#34d399", downColor: "#fb7185", borderVisible: false, wickUpColor: "#34d399", wickDownColor: "#fb7185", priceLineVisible: true, lastValueVisible: true });
  candles.setData(data.candles);
  const volume = chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "volume", lastValueVisible: false, priceLineVisible: false });
  volume.priceScale().applyOptions({ scaleMargins: { top: .82, bottom: 0 } });
  volume.setData(data.volume);
  const line = (values, color, width = 1, style = 0) => {
    const series = chart.addLineSeries({ color, lineWidth: width, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    series.setData(values || []);
    return series;
  };
  line(data.indicators.ema50, "#fbbf24", 2);
  line(data.indicators.ema200, "#60a5fa", 2);
  line(data.indicators.bb_upper, "rgba(167,139,250,.68)", 1);
  line(data.indicators.bb_middle, "rgba(167,139,250,.38)", 1, 2);
  line(data.indicators.bb_lower, "rgba(167,139,250,.68)", 1);
  chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, data.candles.length - 110), to: data.candles.length + 5 });
  container.ondblclick = () => chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, data.candles.length - 110), to: data.candles.length + 5 });
  const observer = new ResizeObserver(() => chart.applyOptions({ width: container.clientWidth, height: container.clientHeight }));
  observer.observe(container);
  state.observer = observer;
  state.chart = chart;
  state.candleSeries = candles;
}

async function loadMeta() {
  try {
    const meta = await requestJson("/api/meta");
    $("featureGrid").innerHTML = meta.feature_groups.map((group) => `<article class="feature-card"><h3>${group.title}</h3><p>${group.plain}</p><div>${group.features.map((feature) => `<code>${feature}</code>`).join("")}</div></article>`).join("");
  } catch (error) { console.warn(error); }
}

async function loadSnapshot() {
  const version = ++state.selectionVersion;
  resetResults();
  updateBadges();
  try {
    const current = selection();
    const data = await requestJson(`/api/snapshot?token=${encodeURIComponent(current.token)}&timeframe=${encodeURIComponent(current.timeframe)}`);
    if (version !== state.selectionVersion) return;
    if (data.validation_summary) renderValidation(data.validation_summary);
    if (!data.has_data) $("chartEmpty").textContent = "저장된 캔들이 없습니다. 데이터 새로고침을 실행하세요.";
    if (data.has_data) {
      await loadChart();
      $("statusTitle").textContent = "저장된 시장 데이터를 불러왔습니다";
      $("statusMessage").textContent = `${fmt.int(data.candle_count)}개 · 마지막 종료 ${fmt.time(data.last_closed_at)} · ${data.is_stale ? "새 캔들 갱신이 필요합니다." : "최신 종료봉이 저장되어 있습니다."} 최신 확률은 버튼을 눌러 확인하세요.`;
    }
  } catch (error) { if (version === state.selectionVersion) fail(error.message); }
}

function setupSidebar() {
  const open = () => { $("sidebar").classList.add("open"); $("backdrop").classList.add("show"); };
  const close = () => { $("sidebar").classList.remove("open"); $("backdrop").classList.remove("show"); };
  $("openSidebar").addEventListener("click", open);
  $("closeSidebar").addEventListener("click", close);
  $("backdrop").addEventListener("click", close);
  window.addEventListener("keydown", (event) => { if (event.key === "Escape") close(); });
}

document.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => startAction(button.dataset.action)));
$("token").addEventListener("change", loadSnapshot);
$("timeframe").addEventListener("change", async () => { updateBadges(); await loadSnapshot(); });
setupSidebar();
loadMeta();
async function initialize() {
  let saved;
  try { saved = JSON.parse(sessionStorage.getItem("activeJob")); } catch (_) {}
  if (saved?.id && saved.token && saved.timeframe) {
    $("token").value = saved.token;
    $("timeframe").value = saved.timeframe;
    updateBadges();
    setBusy(true, "진행 중인 작업에 다시 연결합니다", "서버의 작업 상태를 확인하고 있습니다.");
    try { await pollJob(saved.id); } catch (error) { fail(error.message); }
  } else { await loadSnapshot(); }
}
initialize();

function resetResults() {
  destroyChart();
  $("chartEmpty").style.display = "grid";
  $("chartEmpty").textContent = "저장된 캔들을 확인하고 있습니다.";
  const ids = ["referenceClose", "upProbability", "downProbability", "threshold", "upMini", "downMini", "gaugeValue", "coverage", "pinball", "rocAuc", "brier", "gap", "foldStd", "oosSamples", "foldCount", "trades", "winRate", "return", "mdd", "profitFactor", "sharpe", "q10Price", "q50Price", "q90Price", "q10Return", "q50Return", "q90Return"];
  ids.forEach(id => { $(id).textContent = "—"; $(id).classList.remove("positive", "negative"); });
  setSignal("WAITING");
  $("upFill").style.width = "50%";
  $("needle").style.transform = "rotate(0deg)";
  $("predictionMeta").textContent = "선택한 코인의 최신 예측을 실행하세요.";
  $("directionExplanation").textContent = "예측 대기 중";
  $("overfitStatus").textContent = "검증 결과 없음";
  setBusy(false, "데이터 확인", "최신 예측 또는 데이터 새로고침을 실행하세요.");
}

$("allCandles").addEventListener("click", () => loadChart(100000));
