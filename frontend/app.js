const optionsByClass = {
  Oil: ["BP.L", "SHEL.L"],
  Bullion: ["PHAU.L", "PHGP.L"],
  Forex: ["GBPE.L", "EUGB.L"],
  Stocks: ["VOD.L", "HSBA.L", "BARC.L", "GSK.L"],
};

const seededMetrics = {
  "BP.L": ["0.012", "0.046", "0.008", "0.61"],
  "SHEL.L": ["0.015", "0.051", "0.010", "0.64"],
  "PHAU.L": ["0.010", "0.041", "0.006", "0.49"],
  "PHGP.L": ["0.011", "0.043", "0.006", "0.52"],
  "GBPE.L": ["0.009", "0.039", "0.005", "0.46"],
  "EUGB.L": ["0.010", "0.040", "0.005", "0.47"],
  "VOD.L": ["0.018", "0.058", "0.011", "0.67"],
  "HSBA.L": ["0.017", "0.055", "0.011", "0.63"],
  "BARC.L": ["0.020", "0.061", "0.012", "0.70"],
  "GSK.L": ["0.014", "0.048", "0.009", "0.57"],
};

const assetClassEl = document.getElementById("assetClass");
const instrumentEl = document.getElementById("instrument");

Object.keys(optionsByClass).forEach((assetClass) => {
  const opt = document.createElement("option");
  opt.value = assetClass;
  opt.textContent = assetClass;
  assetClassEl.appendChild(opt);
});

function drawChart(seed = 1) {
  const canvas = document.getElementById("demoChart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#27457f";
  for (let i = 0; i < 6; i++) {
    const y = 20 + i * 40;
    ctx.beginPath();
    ctx.moveTo(20, y);
    ctx.lineTo(canvas.width - 20, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#4f8cff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let x = 0; x < 80; x++) {
    const px = 20 + (x / 79) * (canvas.width - 40);
    const py = 130 + Math.sin((x + seed) / 8) * 32 + Math.cos((x + seed) / 5) * 16;
    if (x === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function populateInstruments() {
  instrumentEl.innerHTML = "";
  const values = optionsByClass[assetClassEl.value];
  values.forEach((symbol) => {
    const opt = document.createElement("option");
    opt.value = symbol;
    opt.textContent = symbol;
    instrumentEl.appendChild(opt);
  });
  updateDashboard();
}

function updateDashboard() {
  const symbol = instrumentEl.value;
  const metrics = seededMetrics[symbol] || ["-", "-", "-", "-"];
  document.getElementById("m1").textContent = metrics[0];
  document.getElementById("m2").textContent = metrics[1];
  document.getElementById("m3").textContent = metrics[2];
  document.getElementById("m4").textContent = metrics[3];
  drawChart(symbol.length * 3);
}

assetClassEl.addEventListener("change", populateInstruments);
instrumentEl.addEventListener("change", updateDashboard);

assetClassEl.value = "Stocks";
populateInstruments();
