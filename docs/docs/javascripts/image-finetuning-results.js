(() => {
  "use strict";

  const DATA_URL = "../assets/data/image-finetuning-results.json";
  const SCHEMA_VERSION = 1;
  const PLOT_CONFIG = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };
  const COLORS = {
    "finetune-lila": "#1f77b4",
    "finetune-speciesnet": "#ff7f0e",
    "scratch-lila-arch": "#2ca02c",
    "scratch-speciesnet-arch": "#d62728",
    "linearprobe-lila": "#9467bd",
    "linearprobe-speciesnet": "#8c564b",
    "balanced-aug-lila": "#17becf",
    "linearprobe-bioclip": "#e377c2",
    "zeroshot-pe-core-g": "#7f7f7f",
    "linearprobe-dinov3-cls": "#bcbd22",
    "linearprobe-dinov3-meanpatch": "#aec7e8",
    "linearprobe-dinov3-concat": "#c49c94",
  };
  const SUMMARY_CONDITIONS = [
    "finetune-lila",
    "finetune-speciesnet",
    "scratch-lila-arch",
    "balanced-aug-lila",
  ];
  const REQUIRED_CONDITIONS = [
    "finetune-lila",
    "finetune-speciesnet",
    "scratch-lila-arch",
    "balanced-aug-lila",
  ];
  const CONDITION_SETS = {
    core: [
      "finetune-lila",
      "finetune-speciesnet",
      "scratch-lila-arch",
      "scratch-speciesnet-arch",
    ],
    sampling: ["finetune-lila", "balanced-aug-lila"],
    probes: [
      "finetune-lila",
      "finetune-speciesnet",
      "linearprobe-lila",
      "linearprobe-speciesnet",
    ],
    external: [
      "finetune-lila",
      "linearprobe-bioclip",
      "linearprobe-dinov3-cls",
      "linearprobe-dinov3-meanpatch",
      "linearprobe-dinov3-concat",
      "zeroshot-pe-core-g",
    ],
  };
  const FREQUENCY_GROUPS = [
    { id: "under50", label: "Fewer than 50", min: 0, max: 50 },
    { id: "under200", label: "50–199", min: 50, max: 200 },
    { id: "under500", label: "200–499", min: 200, max: 500 },
    { id: "common", label: "500 or more", min: 500, max: Infinity },
  ];
  const GROUP_COLORS = {
    under50: "#d62728",
    under200: "#ff7f0e",
    under500: "#1f77b4",
    common: "#7f7f7f",
  };

  const state = {
    data: null,
    pairs: new Map(),
    runSort: { key: "dataset", direction: 1 },
    classSort: { key: "delta", direction: -1 },
    classPage: 0,
    classPageSize: 50,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function number(value, digits = 3) {
    return value === null || value === undefined || Number.isNaN(value)
      ? "—"
      : Number(value).toFixed(digits);
  }

  function signed(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    const text = Number(value).toFixed(digits);
    return Number(value) > 0 ? `+${text}` : text;
  }

  function integer(value) {
    return value === null || value === undefined || Number.isNaN(value)
      ? "—"
      : Number(value).toLocaleString();
  }

  function mean(values) {
    const numbers = values.filter((value) => Number.isFinite(value));
    if (!numbers.length) return null;
    return numbers.reduce((total, value) => total + value, 0) / numbers.length;
  }

  function median(values) {
    const numbers = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
    if (!numbers.length) return null;
    const middle = Math.floor(numbers.length / 2);
    return numbers.length % 2
      ? numbers[middle]
      : (numbers[middle - 1] + numbers[middle]) / 2;
  }

  function datasetLabel(slug) {
    return state.data.dataset_labels[slug] || slug;
  }

  function conditionLabel(slug) {
    return state.data.condition_labels[slug] || slug;
  }

  function frequencyGroup(trainCount) {
    return FREQUENCY_GROUPS.find(
      (group) => trainCount >= group.min && trainCount < group.max,
    );
  }

  function unique(values) {
    return [...new Set(values)];
  }

  function plotlyAvailable() {
    return typeof window.Plotly !== "undefined";
  }

  function showChartUnavailable(id) {
    const chart = byId(id);
    chart.textContent = "Chart unavailable. The equivalent data tables and CSV downloads remain available.";
    chart.classList.add("results-chart-unavailable");
    chart.removeAttribute("role");
  }

  function validateData(data) {
    if (!data || typeof data !== "object") throw new Error("invalid root object");
    if (data.schema_version !== SCHEMA_VERSION) {
      throw new Error(`unsupported schema version ${data.schema_version}`);
    }
    if (!/^[0-9a-f]{40}$/.test(data.source_commit || "")) {
      throw new Error("missing source commit");
    }
    if (!data.methods || !Array.isArray(data.methods.training_tiers)) {
      throw new Error("missing training tiers");
    }
    if (!Array.isArray(data.runs) || !Array.isArray(data.class_results)) {
      throw new Error("missing result tables");
    }
    if (!data.condition_labels || !data.dataset_labels) {
      throw new Error("missing labels");
    }

    const tiers = data.methods.training_tiers;
    if (!tiers.length || tiers.some((tier) => !Number.isFinite(tier) || tier <= 0)) {
      throw new Error("invalid training tiers");
    }

    const runKeys = new Set();
    data.runs.forEach((row) => {
      const key = `${row.dataset}\0${row.condition}\0${row.tier}`;
      if (runKeys.has(key)) throw new Error(`duplicate run ${key.replaceAll("\0", "/")}`);
      runKeys.add(key);
      if (
        !data.dataset_labels[row.dataset] ||
        !data.condition_labels[row.condition] ||
        !tiers.includes(row.tier)
      ) {
        throw new Error(`invalid run identity ${key.replaceAll("\0", "/")}`);
      }
      ["overall_accuracy", "macro_accuracy_reliable_classes"].forEach((metric) => {
        if (!Number.isFinite(row[metric]) || row[metric] < 0 || row[metric] > 1) {
          throw new Error(`invalid ${metric} for ${key.replaceAll("\0", "/")}`);
        }
      });
    });

    Object.keys(data.dataset_labels).forEach((dataset) => {
      tiers.forEach((tier) => {
        REQUIRED_CONDITIONS.forEach((condition) => {
          if (!runKeys.has(`${dataset}\0${condition}\0${tier}`)) {
            throw new Error(`missing run ${dataset}/${condition}/${tier}`);
          }
        });
      });
    });

    const pairKeys = new Map();
    const locationKeys = new Map([
      [10, new Set()],
      [10000, new Set()],
    ]);
    data.class_results.forEach((row) => {
      if (!data.dataset_labels[row.dataset] || !tiers.includes(row.tier)) {
        throw new Error(`invalid class result ${row.dataset}/${row.label}/${row.tier}`);
      }
      if (
        ["finetune-lila", "balanced-aug-lila"].includes(row.condition) &&
        row.quality === "specific"
      ) {
        if (
          !Number.isFinite(row.accuracy) ||
          row.accuracy < 0 ||
          row.accuracy > 1 ||
          !Number.isFinite(row.train_n)
        ) {
          throw new Error(`invalid paired class result ${row.dataset}/${row.label}/${row.tier}`);
        }
        const key = `${row.dataset}\0${row.label}\0${row.tier}`;
        if (!pairKeys.has(key)) pairKeys.set(key, new Set());
        pairKeys.get(key).add(row.condition);
      }
      if (
        locationKeys.has(row.tier) &&
        row.condition === "finetune-lila" &&
        row.genuine &&
        row.reliable &&
        row.det_ok &&
        Number.isFinite(row.n_loc) &&
        Number.isFinite(row.accuracy)
      ) {
        locationKeys.get(row.tier).add(`${row.dataset}\0${row.label}`);
      }
    });
    if (!pairKeys.size || [...pairKeys.values()].some((conditions) => conditions.size !== 2)) {
      throw new Error("missing paired class results");
    }
    if (
      locationKeys.get(10000).size < 3 ||
      [...locationKeys.get(10000)].some((key) => !locationKeys.get(10).has(key))
    ) {
      throw new Error("missing location-comparison class results");
    }
  }

  function setOptions(select, values, labelFunction, includeAll = false) {
    select.replaceChildren();
    if (includeAll) {
      const option = document.createElement("option");
      option.value = "all";
      option.textContent = "All";
      select.append(option);
    }
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = labelFunction(value);
      select.append(option);
    });
  }

  function plotLayout(layout) {
    const styles = getComputedStyle(document.body);
    const color = styles.getPropertyValue("--md-default-fg-color").trim() || "#222";
    const grid = styles.getPropertyValue("--md-default-fg-color--lightest").trim() || "#ddd";
    ["xaxis", "yaxis"].forEach((axis) => {
      layout[axis] = Object.assign(
        {
          gridcolor: grid,
          zerolinecolor: grid,
          tickfont: { color },
          titlefont: { color },
        },
        layout[axis] || {},
      );
    });
    if (layout.title && typeof layout.title === "object") {
      layout.title.font = Object.assign({ color, size: 16 }, layout.title.font || {});
    }
    return Object.assign(
      {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color, size: 12 },
        margin: { t: 50, r: 20, b: 90, l: 65 },
      },
      layout,
    );
  }

  function renderStaticTable(tableId, rows, columns, captionText) {
    const table = byId(tableId);
    table.replaceChildren();
    const caption = document.createElement("caption");
    caption.textContent = captionText;
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    columns.forEach((column) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = column.label;
      headRow.append(cell);
    });
    head.append(headRow);
    const body = document.createElement("tbody");
    rows.forEach((row) => {
      const tableRow = document.createElement("tr");
      columns.forEach((column) => {
        const cell = document.createElement("td");
        cell.textContent = column.format ? column.format(row) : row[column.key];
        tableRow.append(cell);
      });
      body.append(tableRow);
    });
    table.append(caption, head, body);
  }

  function renderSortableTable(tableId, rows, columns, sortState, onSort, captionText) {
    const table = byId(tableId);
    table.replaceChildren();
    const caption = document.createElement("caption");
    caption.textContent = captionText;
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    columns.forEach((column) => {
      const cell = document.createElement("th");
      const button = document.createElement("button");
      const selected = sortState.key === column.key;
      cell.scope = "col";
      cell.setAttribute(
        "aria-sort",
        selected ? (sortState.direction > 0 ? "ascending" : "descending") : "none",
      );
      button.type = "button";
      button.textContent = `${column.label}${
        selected ? (sortState.direction > 0 ? " ▲" : " ▼") : ""
      }`;
      button.setAttribute(
        "aria-label",
        `Sort by ${column.label}${
          selected
            ? sortState.direction > 0
              ? ", currently ascending"
              : ", currently descending"
            : ""
        }`,
      );
      button.addEventListener("click", () => onSort(column.key));
      cell.append(button);
      headRow.append(cell);
    });
    head.append(headRow);
    const body = document.createElement("tbody");
    rows.forEach((row) => {
      const tableRow = document.createElement("tr");
      columns.forEach((column) => {
        const cell = document.createElement("td");
        cell.textContent = column.format
          ? column.format(row[column.key], row)
          : row[column.key] ?? "—";
        tableRow.append(cell);
      });
      body.append(tableRow);
    });
    table.append(caption, head, body);
  }

  function sortRows(rows, sortState) {
    return [...rows].sort((left, right) => {
      const a = left[sortState.key];
      const b = right[sortState.key];
      if (a === b) return 0;
      if (a === null || a === undefined) return 1;
      if (b === null || b === undefined) return -1;
      return (typeof a === "number" ? a - b : String(a).localeCompare(String(b))) *
        sortState.direction;
    });
  }

  function runsFor(condition, tier) {
    return state.data.runs.filter(
      (row) => row.condition === condition && row.tier === tier,
    );
  }

  function classPairs(tier) {
    if (state.pairs.has(tier)) return state.pairs.get(tier);
    const baseline = new Map();
    const balanced = new Map();
    state.data.class_results.forEach((row) => {
      if (row.tier !== tier || row.quality !== "specific" || !Number.isFinite(row.train_n)) return;
      const key = `${row.dataset}\0${row.label}`;
      if (row.condition === "finetune-lila") baseline.set(key, row);
      if (row.condition === "balanced-aug-lila") balanced.set(key, row);
    });
    const pairs = [];
    baseline.forEach((base, key) => {
      const alternative = balanced.get(key);
      if (!alternative) return;
      const group = frequencyGroup(base.train_n);
      pairs.push({
        dataset: base.dataset,
        label: base.label,
        train_n: base.train_n,
        n_loc: base.n_loc,
        support: base.support,
        baseline: base.accuracy,
        balanced: alternative.accuracy,
        delta: alternative.accuracy - base.accuracy,
        group: group.id,
        groupLabel: group.label,
      });
    });
    state.pairs.set(tier, pairs);
    return pairs;
  }

  function speciesnetComparisons(metric = "macro_accuracy_reliable_classes") {
    const tiers = [10, 100, 1000, 10000];
    const datasets = unique(
      state.data.runs
        .filter((row) => row.condition === "finetune-lila")
        .map((row) => row.dataset),
    );
    let wins = 0;
    let total = 0;
    const meanGaps = tiers.map((tier) => {
      const deltas = datasets.map((dataset) => {
        const lila = state.data.runs.find(
          (row) => row.dataset === dataset && row.tier === tier && row.condition === "finetune-lila",
        );
        const species = state.data.runs.find(
          (row) =>
            row.dataset === dataset && row.tier === tier && row.condition === "finetune-speciesnet",
        );
        const delta = lila[metric] - species[metric];
        total += 1;
        if (delta > 0) wins += 1;
        return delta;
      });
      return mean(deltas);
    });
    return { wins, total, meanGaps };
  }

  function macroMean(condition, tier) {
    return mean(runsFor(condition, tier).map((row) => row.macro_accuracy_reliable_classes));
  }

  function updateTakeaways() {
    const finetune10 = macroMean("finetune-lila", 10);
    const finetune100 = macroMean("finetune-lila", 100);
    const finetune1000 = macroMean("finetune-lila", 1000);
    const scratch10 = macroMean("scratch-lila-arch", 10);
    const scratch1000 = macroMean("scratch-lila-arch", 1000);
    const speciesnet = speciesnetComparisons();
    const rarePairs = classPairs(10000).filter((row) => row.train_n < 200);
    const commonPairs = classPairs(10000).filter((row) => row.train_n >= 500);
    const commonMedian = median(commonPairs.map((row) => row.delta));
    const locations = locationGroups();
    byId("takeaway-data").textContent =
      `Fine-tuned lila.science averaged ${number(finetune10, 2)} reliable-class macro accuracy at 10 training images per class and ${number(
        finetune100,
        2,
      )} at 100. It was ${number(finetune1000, 2)} at 1,000, but individual datasets varied substantially.`;
    byId("takeaway-scratch").textContent =
      `The same architecture started from ImageNet reached ${number(scratch10, 2)} at 10 images per class, compared with ${number(
        finetune10,
        2,
      )} for fine-tuning. At 1,000 images per class it was ${number(scratch1000, 2)}, still below fine-tuning at ${number(
        finetune1000,
        2,
      )}.`;
    byId("takeaway-speciesnet-detail").textContent =
      `It had higher macro accuracy in ${speciesnet.wins} of ${speciesnet.total} comparisons on the same data. The mean gap was ${signed(
        speciesnet.meanGaps[0],
        2,
      )} at 10 images per class. lila.science uses 224×224 images. SpeciesNet uses 480×480 images and more GPU memory.`;
    const few = locations[0];
    const many = locations[locations.length - 1];
    byId("takeaway-locations").textContent =
      `At the 10-image training cap, classes in the highest total-catalog-location group averaged ${number(
        many.accuracy10,
        2,
      )} accuracy and classes in the lowest group averaged ${number(
        few.accuracy10,
        2,
      )}.`;
    byId("takeaway-balance").textContent =
      `With inverse-frequency sampling plus stronger augmentation, classes with fewer than 200 training images gained ${signed(
        mean(rarePairs.map((row) => row.delta)),
        2,
      )} at the 10,000-image cap. Classes with 500 or more images changed by ${
        Math.abs(commonMedian) < 0.01 ? "about zero" : signed(commonMedian, 2)
      }.`;
  }

  function updateSummaryScaling() {
    const tiers = [10, 100, 1000, 10000];
    const metric = byId("summary-metric").value;
    const macro = metric === "macro_accuracy_reliable_classes";
    const speciesnet = speciesnetComparisons(metric);
    const traces = SUMMARY_CONDITIONS.map((condition) => ({
      x: tiers,
      y: tiers.map((tier) => mean(runsFor(condition, tier).map((row) => row[metric]))),
      name: conditionLabel(condition),
      type: "scatter",
      mode: "lines+markers",
      line: { color: COLORS[condition], width: 3 },
      marker: { size: 9 },
      hovertemplate:
        `%{x:,} image cap/class<br>mean ${macro ? "macro" : "micro"} accuracy=%{y:.3f}<extra>%{fullData.name}</extra>`,
    }));
    byId("scaling-caption").textContent =
      (macro
        ? "Each point gives the eight datasets equal weight. Within a dataset, macro accuracy gives each reliable class equal weight and excludes classes with fewer than 30 scored test crops. "
        : "Each point gives the eight datasets equal weight. Within a dataset, micro accuracy gives each scored crop equal weight. ") +
      "Fine-tuned lila.science stays above fine-tuned SpeciesNet on average at every cap " +
      `(${speciesnet.meanGaps.map((gap) => signed(gap, 2)).join(", ")}).` +
      (macro
        ? " The result can fall at 10,000 images per class when common classes dominate training."
        : "");
    if (!plotlyAvailable()) {
      showChartUnavailable("summary-scaling-chart");
      return;
    }
    Plotly.react(
      "summary-scaling-chart",
      traces,
      plotLayout({
        title: { text: "Mean result across eight datasets", font: { size: 16 } },
        xaxis: {
          title: "Training-image cap per class",
          type: "log",
          tickvals: tiers,
          ticktext: ["10", "100", "1,000", "10,000"],
        },
        yaxis: { title: macro ? "Mean macro accuracy" : "Mean micro accuracy", range: [0.4, 0.9] },
        legend: { orientation: "h", y: -0.25 },
      }),
      PLOT_CONFIG,
    );
  }

  function groupSummary(tier) {
    const pairs = classPairs(tier);
    return FREQUENCY_GROUPS.map((group) => {
      const rows = pairs.filter((row) => row.group === group.id);
      return {
        label: group.label,
        count: rows.length,
        meanDelta: mean(rows.map((row) => row.delta)),
        medianDelta: median(rows.map((row) => row.delta)),
      };
    });
  }

  function updateSampling() {
    const tiers = [1000, 10000];
    const summaries = Object.fromEntries(tiers.map((tier) => [tier, groupSummary(tier)]));
    const rare10000 = classPairs(10000).filter((row) => row.train_n < 200);
    const middle1000 = summaries[1000][2];
    const middle10000 = summaries[10000][2];
    const common1000 = summaries[1000][3];
    const common10000 = summaries[10000][3];
    byId("sampling-summary").textContent =
      `Specific classes with fewer than 200 training images gained ${signed(
        mean(rare10000.map((row) => row.delta)),
      )} accuracy at the 10,000-image cap (${rare10000.length} classes). ` +
      `Classes with 200–499 images changed by ${signed(middle1000.meanDelta)} at the 1,000 cap and ${signed(
        middle10000.meanDelta,
      )} at the 10,000 cap. ` +
      `Classes with 500 or more images changed by ${signed(common1000.meanDelta)} at the 1,000 cap and ${signed(
        common10000.meanDelta,
      )} at the 10,000 cap. Their median change was ${signed(common1000.medianDelta)} and ${signed(
        common10000.medianDelta,
      )}.`;

    if (plotlyAvailable()) {
      Plotly.react(
        "sampling-group-chart",
        tiers.map((tier, index) => ({
          x: summaries[tier].map((group) => group.label),
          y: summaries[tier].map((group) => group.meanDelta),
          name: `${tier.toLocaleString()} image cap`,
          type: "bar",
          marker: { color: index === 0 ? "#8ecae6" : "#17becf" },
          customdata: summaries[tier].map((group) => [group.count, group.medianDelta]),
          hovertemplate:
            "%{x}<br>mean change=%{y:.3f}<br>median=%{customdata[1]:.3f}" +
            "<br>classes=%{customdata[0]}<extra>%{fullData.name}</extra>",
        })),
        plotLayout({
          barmode: "group",
          title: { text: "Mean accuracy change by class frequency", font: { size: 16 } },
          xaxis: { title: "Actual training images" },
          yaxis: { title: "Combined setting minus baseline", zerolinewidth: 2 },
          legend: { orientation: "h", y: -0.25 },
        }),
        PLOT_CONFIG,
      );
    } else {
      showChartUnavailable("sampling-group-chart");
    }

    renderStaticTable(
      "sampling-group-table",
      FREQUENCY_GROUPS.map((group, index) => ({
        label: group.label,
        count: summaries[10000][index].count,
        low: summaries[1000][index].meanDelta,
        high: summaries[10000][index].meanDelta,
        median: summaries[10000][index].medianDelta,
      })),
      [
        { key: "label", label: "Training images" },
        { key: "count", label: "Classes", format: (row) => integer(row.count) },
        { key: "low", label: "Mean change at 1,000", format: (row) => signed(row.low) },
        { key: "high", label: "Mean change at 10,000", format: (row) => signed(row.high) },
        { key: "median", label: "Median change at 10,000", format: (row) => signed(row.median) },
      ],
      "Mean per-class accuracy change for the combined balanced-sampling-plus-augmentation setting versus the standard fine-tune.",
    );
    updateRareGains();
  }

  function updateRareGains() {
    const tier = Number(byId("rare-gain-tier").value);
    const pairs = classPairs(tier);
    const rare = pairs
      .filter((row) => row.train_n < 200)
      .sort((a, b) => b.delta - a.delta);
    renderStaticTable(
      "rare-gain-table",
      rare.slice(0, 12),
      [
        { key: "label", label: "Class" },
        { key: "dataset", label: "Dataset", format: (row) => datasetLabel(row.dataset) },
        { key: "train_n", label: "Train images", format: (row) => integer(row.train_n) },
        { key: "support", label: "Test images", format: (row) => integer(row.support) },
        { key: "baseline", label: "Baseline", format: (row) => number(row.baseline) },
        { key: "balanced", label: "Balanced + aug.", format: (row) => number(row.balanced) },
        { key: "delta", label: "Change", format: (row) => signed(row.delta) },
      ],
      "The 12 largest gains among specific classes with fewer than 200 training images.",
    );

    const traces = FREQUENCY_GROUPS.map((group) => {
      const rows = pairs.filter((row) => row.group === group.id);
      return {
        x: rows.map((row) => row.train_n),
        y: rows.map((row) => row.delta),
        name: group.label,
        type: "scatter",
        mode: "markers",
        marker: { color: GROUP_COLORS[group.id], size: 8, opacity: 0.75 },
        customdata: rows.map((row) => [row.label, datasetLabel(row.dataset), row.support]),
        hovertemplate:
          "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>" +
          "train images=%{x:,}<br>change=%{y:.3f}<br>test images=%{customdata[2]:,}<extra></extra>",
      };
    });
    if (!plotlyAvailable()) {
      showChartUnavailable("sampling-class-chart");
      return;
    }
    Plotly.react(
      "sampling-class-chart",
      traces,
      plotLayout({
        title: { text: "Every specific class", font: { size: 16 } },
        xaxis: { title: "Actual training images", type: "log" },
        yaxis: { title: "Combined setting minus baseline", zerolinewidth: 2 },
        legend: { orientation: "h", y: -0.25 },
        hovermode: "closest",
      }),
      PLOT_CONFIG,
    );
  }

  function quantile(sortedValues, probability) {
    const position = (sortedValues.length - 1) * probability;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) return sortedValues[lower];
    return sortedValues[lower] * (upper - position) + sortedValues[upper] * (position - lower);
  }

  function screenedLocationRows(tier) {
    return state.data.class_results.filter(
      (row) =>
        row.condition === "finetune-lila" &&
        row.tier === tier &&
        row.genuine &&
        row.reliable &&
        row.det_ok &&
        Number.isFinite(row.n_loc) &&
        Number.isFinite(row.accuracy),
    );
  }

  function locationGroups() {
    const rows = screenedLocationRows(10000);
    const tierRows = new Map(
      [10, 10000].map((tier) => {
        const matches = new Map();
        screenedLocationRows(tier).forEach((row) => {
          matches.set(`${row.dataset}\0${row.label}`, row);
        });
        return [tier, matches];
      }),
    );
    const locations = rows.map((row) => row.n_loc).sort((a, b) => a - b);
    const low = quantile(locations, 1 / 3);
    const high = quantile(locations, 2 / 3);
    const labels = ["Few catalog locations", "Medium coverage", "Many catalog locations"];
    return labels.map((label, index) => {
      const group = rows
        .filter((row) => {
          if (index === 0) return row.n_loc <= low;
          if (index === 1) return row.n_loc > low && row.n_loc <= high;
          return row.n_loc > high;
        })
        .sort((a, b) => a.n_loc - b.n_loc);
      const accuracyAt = (tier) =>
        mean(
          group.map((row) => tierRows.get(tier).get(`${row.dataset}\0${row.label}`)?.accuracy),
        );
      return {
        label,
        accuracy: accuracyAt(10000),
        accuracy10: accuracyAt(10),
        accuracy10000: accuracyAt(10000),
        count: group.length,
        minLocations: group[0]?.n_loc,
        maxLocations: group[group.length - 1]?.n_loc,
      };
    });
  }

  function updateLocations() {
    const groups = locationGroups();
    const few = groups[0];
    const many = groups[groups.length - 1];
    byId("location-summary").textContent =
      `At the 10-image training cap, classes in the highest total-catalog-location group averaged ${number(
        many.accuracy10,
        2,
      )} accuracy and classes in the lowest group averaged ${number(
        few.accuracy10,
        2,
      )} (${integer(few.minLocations)}–${integer(few.maxLocations)} catalog locations versus ${integer(
        many.minLocations,
      )}–${integer(many.maxLocations)}). At the 10,000-image cap, those groups averaged ${number(
        many.accuracy10000,
        2,
      )} and ${number(
        few.accuracy10000,
        2,
      )}, respectively.`;
    const locationHover =
      "%{x}<br>mean accuracy=%{y:.3f}<br>classes=%{customdata[0]}" +
      "<br>catalog locations=%{customdata[1]:,}–%{customdata[2]:,}<extra>%{fullData.name}</extra>";
    const customdata = groups.map((group) => [
      group.count,
      group.minLocations,
      group.maxLocations,
    ]);
    if (!plotlyAvailable()) {
      showChartUnavailable("location-chart");
      return;
    }
    Plotly.react(
      "location-chart",
      [
        {
          x: groups.map((group) => group.label),
          y: groups.map((group) => group.accuracy10),
          name: "10 images per class",
          type: "bar",
          marker: { color: "#8ecae6" },
          customdata,
          hovertemplate: locationHover,
        },
        {
          x: groups.map((group) => group.label),
          y: groups.map((group) => group.accuracy10000),
          name: "10,000 images per class",
          type: "bar",
          marker: { color: "#1f77b4" },
          customdata,
          hovertemplate: locationHover,
        },
      ],
      plotLayout({
        barmode: "group",
        title: {
          text: "Fine-tuned lila.science, grouped by full-catalog coverage",
          font: { size: 16 },
        },
        xaxis: { title: "Total catalog-location groups" },
        yaxis: { title: "Mean class accuracy", range: [0, 1] },
        legend: { orientation: "h", y: -0.25 },
      }),
      PLOT_CONFIG,
    );
  }

  function selectedClassRows() {
    const dataset = byId("class-table-dataset").value;
    const tier = Number(byId("class-table-tier").value);
    const frequency = byId("class-table-frequency").value;
    const query = byId("class-table-search").value.trim().toLowerCase();
    return classPairs(tier).filter((row) => {
      const searchable = `${row.label} ${row.dataset} ${datasetLabel(row.dataset)}`.toLowerCase();
      const frequencyMatch =
        frequency === "all" ||
        (frequency === "rare" && row.train_n < 200) ||
        (frequency === "middle" && row.train_n >= 200 && row.train_n < 500) ||
        (frequency === "common" && row.train_n >= 500);
      return (
        (dataset === "all" || row.dataset === dataset) &&
        frequencyMatch &&
        (!query || searchable.includes(query))
      );
    });
  }

  function renderClassTable() {
    const columns = [
      { key: "label", label: "Class" },
      { key: "dataset", label: "Dataset", format: (value) => datasetLabel(value) },
      { key: "train_n", label: "Train images", format: integer },
      { key: "n_loc", label: "Catalog locations", format: integer },
      { key: "support", label: "Test images", format: integer },
      { key: "baseline", label: "Baseline", format: number },
      { key: "balanced", label: "Balanced + aug.", format: number },
      { key: "delta", label: "Change", format: signed },
    ];
    const filtered = sortRows(selectedClassRows(), state.classSort);
    const pageCount = Math.max(1, Math.ceil(filtered.length / state.classPageSize));
    state.classPage = Math.min(state.classPage, pageCount - 1);
    const start = state.classPage * state.classPageSize;
    renderSortableTable(
      "class-results-table",
      filtered.slice(start, start + state.classPageSize),
      columns,
      state.classSort,
      (key) => {
        state.classSort.direction = state.classSort.key === key ? state.classSort.direction * -1 : -1;
        state.classSort.key = key;
        state.classPage = 0;
        renderClassTable();
      },
      "Per-class standard fine-tune versus combined balanced-sampling-plus-augmentation results.",
    );
    byId("class-table-count").textContent =
      `${filtered.length.toLocaleString()} classes · page ${state.classPage + 1} of ${pageCount}`;
    byId("class-page-previous").disabled = state.classPage === 0;
    byId("class-page-next").disabled = state.classPage >= pageCount - 1;
  }

  function updateDetailChart() {
    const dataset = byId("scaling-dataset").value;
    const metric = byId("scaling-metric").value;
    const conditions = CONDITION_SETS[byId("scaling-conditions").value];
    const missing = conditions.filter(
      (condition) =>
        !state.data.runs.some((row) => row.dataset === dataset && row.condition === condition),
    );
    byId("detail-chart-note").textContent = missing.length
      ? `No completed runs for this dataset: ${missing.map(conditionLabel).join(", ")}.`
      : "";
    const traces = conditions
      .map((condition) => {
        const rows = state.data.runs
          .filter((row) => row.dataset === dataset && row.condition === condition)
          .sort((a, b) => a.tier - b.tier);
        return {
          x: rows.map((row) => row.tier),
          y: rows.map((row) => row[metric]),
          name: conditionLabel(condition),
          type: "scatter",
          mode: "lines+markers",
          line: { color: COLORS[condition], width: 2 },
          marker: { size: 8 },
          hovertemplate: "%{x:,} image cap/class<br>%{y:.3f}<extra>%{fullData.name}</extra>",
        };
      })
      .filter((trace) => trace.x.length);
    if (!plotlyAvailable()) {
      showChartUnavailable("detail-scaling-chart");
      return;
    }
    Plotly.react(
      "detail-scaling-chart",
      traces,
      plotLayout({
        title: { text: datasetLabel(dataset), font: { size: 16 } },
        xaxis: {
          title: "Training-image cap per class",
          type: "log",
          tickvals: [10, 100, 1000, 10000],
          ticktext: ["10", "100", "1,000", "10,000"],
        },
        yaxis: { title: byId("scaling-metric").selectedOptions[0].text, range: [0, 1] },
        legend: { orientation: "h", y: -0.28 },
      }),
      PLOT_CONFIG,
    );
  }

  function selectedRunRows() {
    const dataset = byId("run-table-dataset").value;
    const condition = byId("run-table-condition").value;
    const tier = byId("run-table-tier").value;
    const query = byId("run-table-search").value.trim().toLowerCase();
    return state.data.runs.filter((row) => {
      const searchable = `${row.dataset} ${datasetLabel(row.dataset)} ${row.condition} ${conditionLabel(row.condition)}`.toLowerCase();
      return (
        (dataset === "all" || row.dataset === dataset) &&
        (condition === "all" || row.condition === condition) &&
        (tier === "all" || row.tier === Number(tier)) &&
        (!query || searchable.includes(query))
      );
    });
  }

  function renderRunTable() {
    const columns = [
      { key: "dataset", label: "Dataset", format: (value) => datasetLabel(value) },
      { key: "condition", label: "Condition", format: (value) => conditionLabel(value) },
      { key: "tier", label: "Train cap", format: integer },
      { key: "overall_accuracy", label: "Overall accuracy", format: number },
      { key: "macro_accuracy_reliable_classes", label: "Macro accuracy", format: number },
      { key: "n_scored_crops", label: "Scored crops", format: integer },
    ];
    const rows = sortRows(selectedRunRows(), state.runSort);
    renderSortableTable(
      "run-results-table",
      rows,
      columns,
      state.runSort,
      (key) => {
        state.runSort.direction = state.runSort.key === key ? state.runSort.direction * -1 : 1;
        state.runSort.key = key;
        renderRunTable();
      },
      "All completed result rows matching the selected filters.",
    );
    byId("run-table-count").textContent = `${rows.length} runs`;
  }

  function bind(ids, render, resetPage = false) {
    ids.forEach((id) => {
      const element = byId(id);
      element.addEventListener(element.tagName === "INPUT" ? "input" : "change", () => {
        if (resetPage) state.classPage = 0;
        render();
      });
    });
  }

  function initializeControls() {
    const datasets = unique(state.data.runs.map((row) => row.dataset)).sort();
    const conditions = unique(state.data.runs.map((row) => row.condition)).sort();
    const tiers = unique(state.data.runs.map((row) => row.tier)).sort((a, b) => a - b);
    setOptions(byId("scaling-dataset"), datasets, datasetLabel);
    setOptions(byId("run-table-dataset"), datasets, datasetLabel, true);
    setOptions(byId("run-table-condition"), conditions, conditionLabel, true);
    setOptions(byId("run-table-tier"), tiers, (value) => Number(value).toLocaleString(), true);
    setOptions(byId("class-table-dataset"), datasets, datasetLabel, true);
    setOptions(byId("class-table-tier"), tiers, (value) => Number(value).toLocaleString());
    byId("class-table-tier").value = "10000";
    bind(["summary-metric"], updateSummaryScaling);
    bind(["scaling-dataset", "scaling-metric", "scaling-conditions"], updateDetailChart);
    bind(["run-table-dataset", "run-table-condition", "run-table-tier", "run-table-search"], renderRunTable);
    bind(
      ["class-table-dataset", "class-table-tier", "class-table-frequency", "class-table-search"],
      renderClassTable,
      true,
    );
    byId("rare-gain-tier").addEventListener("change", updateRareGains);
    document.querySelector(".results-details").addEventListener("toggle", (event) => {
      if (event.target.open && plotlyAvailable()) {
        Plotly.Plots.resize("detail-scaling-chart");
      }
    });
    byId("class-page-previous").addEventListener("click", () => {
      state.classPage -= 1;
      renderClassTable();
    });
    byId("class-page-next").addEventListener("click", () => {
      state.classPage += 1;
      renderClassTable();
    });
  }

  async function initialize() {
    const status = byId("finetuning-results-status");
    try {
      const response = await fetch(DATA_URL);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      validateData(data);
      state.data = data;
      byId("finetuning-source-commit").textContent = state.data.source_commit.slice(0, 12);
      initializeControls();
      updateTakeaways();
      updateSummaryScaling();
      updateSampling();
      updateLocations();
      renderClassTable();
      updateDetailChart();
      renderRunTable();
      if (plotlyAvailable()) {
        status.remove();
      } else {
        status.textContent =
          "Tables loaded. Charts are unavailable because the chart library did not load.";
      }
    } catch (error) {
      status.textContent = `The data snapshot could not load (${error.message}).`;
      status.classList.add("results-error");
    }
  }

  initialize();
})();
