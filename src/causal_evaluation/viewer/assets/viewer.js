const RECORDS = __RECORDS__;
const SUPPORTED_LAYOUTS = __SUPPORTED__;
const byId = new Map(RECORDS.map(r => [r.question_id, r]));

const datalist = document.getElementById("qids");
for (const r of RECORDS) {
  const opt = document.createElement("option");
  opt.value = r.question_id;
  opt.label = `${r.question_id} - ${r.story_id || "?"} - ${r.topology}`;
  datalist.appendChild(opt);
}

let network = null;

function roleColor(varName, record) {
  if (varName === record.treatment) return { bg: "#4E79A7", border: "#7BA7D9" };
  if (varName === record.outcome) return { bg: "#59A14F", border: "#8BC781" };
  return { bg: "#F28E2B", border: "#F5B26B" };
}

function renderGraph(record) {
  const nodeIds = Object.keys(record.variables);
  const hasLayout = record.layout && SUPPORTED_LAYOUTS.includes(record.topology);
  const notice = document.getElementById("notice");
  if (!hasLayout) {
    notice.style.display = "block";
    notice.textContent = `No custom diagram yet for topology "${record.topology}" - showing auto-layout. Currently supported: ${SUPPORTED_LAYOUTS.join(", ")}.`;
  } else {
    notice.style.display = "none";
  }

  const nodes = new vis.DataSet(nodeIds.map(v => {
    const c = roleColor(v, record);
    const base = { id: v, label: `${v}\n${record.variables[v]}`, color: { background: c.bg, border: c.border }, font: { color: "#fff" }, shape: "box", margin: 10 };
    if (hasLayout && record.layout[v]) {
      base.x = record.layout[v][0];
      base.y = record.layout[v][1];
      base.fixed = true;
    }
    return base;
  }));

  const edges = new vis.DataSet(record.edges.map(([from, to]) => ({
    from, to, arrows: "to", color: { color: "#5a5a7e" },
    smooth: hasLayout ? false : { type: "dynamic" }
  })));

  const container = document.getElementById("graph");
  const options = {
    physics: !hasLayout,
    interaction: { hover: true, dragNodes: true, zoomView: true },
    nodes: { borderWidth: 2, shadow: true, font: { size: 14, multi: true } },
    edges: { width: 2 }
  };
  if (network) network.destroy();
  network = new vis.Network(container, { nodes, edges }, options);
  network.on("click", (params) => showNodeInfo(params, record));
}

function showNodeInfo(params, record) {
  const info = document.getElementById("info-content");
  if (params.nodes.length > 0) {
    const id = params.nodes[0];
    const role = id === record.treatment ? "Treatment" : id === record.outcome ? "Outcome" : "Variable";
    info.innerHTML = `
      <div class="field"><b>Variable</b><span>${id} <span class="node-role" style="background:#ffffff22">${role}</span></span></div>
      <div class="field"><b>Meaning</b><span>${record.variables[id] || ""}</span></div>
    `;
  } else if (params.edges.length > 0) {
    info.innerHTML = '<div class="field"><b>Relation</b><span>causal edge</span></div>';
  } else {
    info.innerHTML = '<div class="empty">Click a node to see details</div>';
  }
}

function renderSidebar(record) {
  document.getElementById("q-body").innerHTML = `
    <div class="field"><b>Question</b><span>${record.question}</span></div>
    <div class="field"><b>Given Info</b><span>${record.given_info}</span></div>
  `;
  document.getElementById("q-meta").innerHTML = `
    <div class="field"><b>Topology</b><span>${record.topology}</span></div>
    <div class="field"><b>Query Type</b><span>${record.query_type}</span></div>
    <div class="field"><b>Causal Rung</b><span>${record.causal_level}</span></div>
    <div class="field"><b>Expected Answer</b><span>${record.answer}</span></div>
  `;
  const badgeClass = record.correct === true ? "correct" : record.correct === false ? "incorrect" : "unknown";
  const badgeText = record.correct === true ? "correct" : record.correct === false ? "incorrect" : "no run yet";
  const reasoning = record.reasoning_text
    ? `<details><summary>Show reasoning</summary><div class="field"><span>${record.reasoning_text}</span></div></details>`
    : "";
  document.getElementById("q-model").innerHTML = `
    <div class="field"><b>Model</b><span>${record.model || "-"}</span></div>
    <div class="field"><b>Answer</b><span>${record.response_text || record.response_error || "-"}</span></div>
    <div class="field"><b>Result</b><span class="badge ${badgeClass}">${badgeText}</span></div>
    ${reasoning}
  `;
}

function selectQuestion(id) {
  const record = byId.get(id);
  if (!record) return;
  renderGraph(record);
  renderSidebar(record);
}

document.getElementById("picker").addEventListener("change", (e) => selectQuestion(e.target.value));
document.getElementById("picker").addEventListener("input", (e) => { if (byId.has(e.target.value)) selectQuestion(e.target.value); });

if (RECORDS.length > 0) {
  document.getElementById("picker").value = RECORDS[0].question_id;
  selectQuestion(RECORDS[0].question_id);
}
