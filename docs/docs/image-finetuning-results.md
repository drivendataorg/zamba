# Image fine-tuning experiment results

We fine-tuned image classifiers on eight camera-trap datasets from [LILA BC](https://lila.science/datasets). None of these datasets was used to train the `lila.science` base model that ships with `zamba`, so the results evaluate fine-tuning on new data. Many of the species were already in that base model's training data, and seven of the eight datasets come from continents that training data already covered.

Each camera location was placed in only one split: training, validation, or test. Within a dataset, the same classes were trained four times, with at most 10, 100, 1,000, and 10,000 **training images** per class. Smaller caps are subsets of the larger ones; validation and test images are additional. Scores use MegaDetector animal crops from the held-out locations.

The main comparisons are fine-tuning `lila.science`, fine-tuning SpeciesNet, training those architectures from ImageNet weights, and a combined class-balanced-sampling-plus-augmentation setting. We also trained linear probes (a small linear classifier on top of a frozen base model) for `lila.science` and SpeciesNet on every dataset. On two datasets we used DINOv3 and BioCLIP as frozen feature extractors with the same kind of classifier, and we scored a zero-shot vision-language model that used text prompts and no training images.

The headline macro value first averages qualifying classes within each dataset, excluding classes with fewer than 30 scored test crops, and then gives each of the eight dataset-level values equal weight. The study used one training seed, nested training sets, and shared holdouts, so comparisons across tiers are correlated and do not include training-to-training uncertainty. Charts for individual datasets, full tables, downloads, and the exact benchmark recipe are below. Supported practical steps are in the [fine-tuning tutorial](images-finetune-guide.md).

<p id="finetuning-results-status" role="status" aria-live="polite">Loading charts and tables…</p>

## Takeaways

<div class="takeaway-grid">
  <article>
    <h3>10–100 training images per class is a useful start</h3>
    <p id="takeaway-data">Fine-tuned <code>lila.science</code> averaged 0.74 reliable-class macro accuracy at 10 training images per class and 0.78 at 100. Individual datasets varied substantially.</p>
  </article>
  <article>
    <h3>An ImageNet start has lower accuracy</h3>
    <p id="takeaway-scratch">The same architecture started from ImageNet reached 0.55 at 10 images per class, compared with 0.74 for fine-tuning. At 1,000 images per class it was 0.74, still below fine-tuning at 0.78.</p>
  </article>
  <article>
    <h3><code>lila.science</code> wins more often and uses smaller inputs</h3>
    <p id="takeaway-speciesnet-detail">It had higher macro accuracy in 30 of 32 comparisons on the same data. The mean gap was +0.10 at 10 images per class. <code>lila.science</code> uses 224×224 images. SpeciesNet uses 480×480 images and more GPU memory.</p>
  </article>
  <article>
    <h3>Catalog location coverage is associated with accuracy</h3>
    <p id="takeaway-locations">At the 10-image training cap, classes in the highest total-location group averaged 0.84 accuracy and classes in the lowest group averaged 0.71.</p>
  </article>
  <article>
    <h3>The combined balanced setting helps rare classes</h3>
    <p id="takeaway-balance">With inverse-frequency sampling and stronger augmentation, classes with fewer than 200 training images gained +0.13 at the 10,000-image cap. Classes with 500 or more images changed by about zero.</p>
  </article>
</div>

!!! note "When fine-tuning does poorly"
    There is some initial evidence that a strong frozen feature extractor, or a general vision-language model, is worth comparing when ordinary fine-tuning does poorly. These runs cover two datasets. On California Small Animals, fine-tuned `lila.science` stayed near 0.35–0.46 macro accuracy, and DINOv3 features were higher at every training cap. On the Felidae dataset, a zero-shot vision-language model scored 0.72 and matched or beat fine-tuning. Select "External frozen and zero-shot models" in the per-dataset charts to see the numbers.

## Data scaling across all datasets

<div class="results-controls">
  <label>Accuracy
    <select id="summary-metric">
      <option value="macro_accuracy_reliable_classes" selected>Macro accuracy</option>
      <option value="overall_accuracy">Micro accuracy</option>
    </select>
  </label>
</div>

<p id="scaling-caption" class="results-caption">Each point gives every dataset equal weight. Within a dataset, macro accuracy gives each reliable class equal weight; micro accuracy gives each scored crop equal weight.</p>

<div id="summary-scaling-chart" class="results-chart" role="img" aria-label="Mean accuracy by training data tier" aria-describedby="scaling-caption"></div>

## The combined balanced setting helped rare classes

<p id="sampling-summary">Inverse-frequency sampling plus stronger augmentation gave rare classes a large accuracy gain. Well-represented classes changed very little.</p>

<div id="sampling-group-chart" class="results-chart" role="img" aria-label="Accuracy change by training-image frequency" aria-describedby="sampling-summary"></div>

<div class="results-table-wrap" role="region" aria-label="Balanced-setting summary table" tabindex="0">
  <table id="sampling-group-table" class="results-table"></table>
</div>

<h3>Rare classes with the largest gains</h3>

<div class="results-controls">
  <label>Training-image cap
    <select id="rare-gain-tier">
      <option value="1000">1,000</option>
      <option value="10000" selected>10,000</option>
    </select>
  </label>
</div>

<p id="rare-gain-caption" class="results-caption">Specific classes with fewer than 200 actual training images. The table shows the 12 largest gains at the selected cap. A large change with few test images is less stable.</p>

<div class="results-table-wrap" role="region" aria-label="Largest rare-class gains table" tabindex="0">
  <table id="rare-gain-table" class="results-table"></table>
</div>

<div id="sampling-class-chart" class="results-chart" role="img" aria-label="Accuracy change for each class" aria-describedby="rare-gain-caption"></div>

## Accuracy by total catalog location coverage

<p id="location-summary">Classes present at more locations across the full source catalog had higher average accuracy.</p>

<div id="location-chart" class="results-chart" role="img" aria-label="Accuracy by total catalog locations at two training caps" aria-describedby="location-summary"></div>

## Compare a class across both training settings

<p id="class-table-caption" class="results-caption">Each row is one specific class. Baseline is the standard <code>lila.science</code> fine-tune. Balanced plus augmentation is the combined setting on the same data. “Catalog locations” covers the full source catalog, not training locations at that tier.</p>

<div class="results-controls">
  <label>Dataset
    <select id="class-table-dataset"></select>
  </label>
  <label>Training-image cap
    <select id="class-table-tier"></select>
  </label>
  <label>Class frequency
    <select id="class-table-frequency">
      <option value="all">All classes</option>
      <option value="rare">Rare: fewer than 200 images</option>
      <option value="middle">Less rare: 200–499 images</option>
      <option value="common">Well represented: 500 or more</option>
    </select>
  </label>
  <label>Class search
    <input id="class-table-search" type="search" placeholder="Species or dataset">
  </label>
</div>

<div class="results-pager">
  <span id="class-table-count" class="results-count" role="status" aria-live="polite"></span>
  <span>
    <button id="class-page-previous" class="md-button" type="button">Previous</button>
    <button id="class-page-next" class="md-button" type="button">Next</button>
  </span>
</div>
<div class="results-table-wrap" role="region" aria-label="Per-class comparison table" tabindex="0">
  <table id="class-results-table" class="results-table"></table>
</div>

## Explore individual datasets

<details class="results-details" open>
  <summary>Per-dataset charts, all runs, and downloads</summary>

  <div class="results-controls">
    <label>Dataset
      <select id="scaling-dataset"></select>
    </label>
    <label>Metric
      <select id="scaling-metric">
        <option value="macro_accuracy_reliable_classes">Macro accuracy, reliable classes</option>
        <option value="overall_accuracy">Overall accuracy</option>
        <option value="macro_f1">Macro F1</option>
      </select>
    </label>
    <label>Comparison
      <select id="scaling-conditions">
        <option value="core">Fine-tune versus ImageNet start</option>
        <option value="sampling">Standard versus balanced sampling + augmentation</option>
        <option value="probes">Full fine-tune versus linear probe</option>
        <option value="external">External frozen and zero-shot models</option>
      </select>
    </label>
  </div>
  <p id="detail-chart-note" class="results-caption" role="status" aria-live="polite"></p>
  <div id="detail-scaling-chart" class="results-chart" role="img" aria-label="Accuracy for one dataset" aria-describedby="detail-chart-note"></div>

  <div class="results-controls">
    <label>Dataset
      <select id="run-table-dataset"></select>
    </label>
    <label>Condition
      <select id="run-table-condition"></select>
    </label>
    <label>Training-image cap
      <select id="run-table-tier"></select>
    </label>
    <label>Search
      <input id="run-table-search" type="search" placeholder="Dataset or condition">
    </label>
  </div>
  <p id="run-table-count" class="results-count" role="status" aria-live="polite"></p>
  <div class="results-table-wrap" role="region" aria-label="Run results table" tabindex="0">
    <table id="run-results-table" class="results-table"></table>
  </div>

  <div class="results-downloads">
    <a class="md-button" href="../assets/data/image-finetuning-runs.csv" download>Run results CSV</a>
    <a class="md-button" href="../assets/data/image-finetuning-class-results.csv" download>Class results CSV</a>
    <a class="md-button" href="../assets/data/image-finetuning-datasets.csv" download>Dataset summary CSV</a>
  </div>
</details>

## Methods and definitions

### Data and evaluation

* **Data:** Eight camera-trap datasets from [LILA BC](https://lila.science/datasets). Each class required at least three camera locations, 10 training images, and 10 test images.
* **Splits:** A camera location did not occur in more than one split. The target split of locations was 60% training, 15% validation, and 25% test.
* **Tiers:** Training sets used nested caps of 10, 100, 1,000, and 10,000 training images per class. A rare class could have fewer images than the cap. Validation and test images were separate from these counts.
* **Inputs:** Precomputed detections supplied one crop per selected image: the highest-confidence MegaDetector animal box, or a full-frame fallback when no animal was detected. Classification metrics include only held-out images with a detected animal, so detector misses are counted separately rather than scored.
* **Headline macro accuracy:** Within each dataset, the reliable-class metric is the unweighted mean of class accuracies after excluding classes with fewer than 30 scored test crops. A chart point is then the unweighted mean of those eight dataset-level values. It is not a global average in which every class across all datasets has equal weight.
* **Specific classes:** The combined-setting comparison keeps labels marked specific and removes conflicting labels and broad catch-all labels. The change is combined-setting accuracy minus standard-fine-tune accuracy.
* **Location groups:** Quality-screened classes have a specific species label, at least 30 scored test crops, and at least an 85% MegaDetector detection rate. Groups use the number of distinct locations where a class occurs in the **full source catalog**.
* **Intervals:** Class accuracy intervals in the download are 95% Wilson score intervals for held-out crop outcomes. They do not cover training-seed or dataset-to-dataset uncertainty.

### Training recipe

* **Software:** The locked environment used Zamba 2.7.2. Experiment logic and snapshot provenance are recorded by the source commit shown in the citation below.
* **Optimization:** Full fine-tunes used AdamW with weight decay 0.01. Learning rates were fixed at `1e-4` for both wildlife-pretrained models, `4e-4` for the `lila.science` architecture from ImageNet, and `6e-4` for the SpeciesNet architecture from ImageNet. A three-epoch linear warmup was followed by cosine decay over at most 40 epochs, with early-stopping patience 5. Stochastic weight averaging was disabled.
* **Batches:** `lila.science` used 224×224 inputs and batch size 32. SpeciesNet used 480×480 inputs and batch size 8. The benchmark did not use gradient accumulation.
* **Combined balanced condition:** `balanced-aug-lila` changed two things together: inverse-frequency sampling with replacement and stronger camera-trap augmentations. The results do not identify the effect of either component alone.
* **Linear probes:** Zamba probes froze the backbone and trained only its classification head. BioCLIP and DINOv3 used frozen features with multinomial logistic regression.
* **Zero shot:** `zeroshot-pe-core-g` used text prompts and no training images. Each of its two distinct evaluations is repeated at all four displayed tiers, so the snapshot has 264 result rows representing 258 distinct evaluations.
* **Uncertainty:** Each trained condition used one seed (`42`). Tiers are nested and reuse the same holdout, so tier comparisons are correlated. No repeated-seed interval is available.

These settings differ from the supported defaults in the [practical tutorial](images-finetune-guide.md); its commands are not a reproduction script.

## Citations

Suggested citation for this evidence page:

> DrivenData (2026). *Image fine-tuning experiment results*. Zamba documentation. Experiment snapshot <code id="finetuning-source-commit">85d1a6bb3191</code>.

Model, data, and method sources:

1. Dorne, E., Qi, J., Bull, P., et al. (2025). [Zamba: Computer vision for wildlife conservation](https://doi.org/10.25080/crcw9835). *Proceedings of the Python in Science Conference*.
2. Morris, D. [Labeled Information Library of Alexandria: Biology and Conservation](https://lila.science/). Dataset-specific citations and licenses are available in the [LILA dataset catalog](https://lila.science/datasets).
3. Beery, S., Van Horn, G., and Perona, P. (2018). [Recognition in Terra Incognita](https://doi.org/10.1007/978-3-030-01270-0_28). *ECCV 2018*. This paper motivates evaluation at camera locations that are not in the training data.
4. Beery, S., Morris, D., and Yang, S. (2019). [Efficient Pipeline for Camera Trap Image Review](https://arxiv.org/abs/1907.06772). arXiv:1907.06772. This is the MegaDetector citation.
5. Gadot, T., Istrate, Ș., Kim, H., et al. (2024). [To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images](https://doi.org/10.1049/cvi2.12318). *IET Computer Vision*, 18(8), 1193–1208. This is the SpeciesNet citation.
6. Wilson, E. B. (1927). [Probable inference, the law of succession, and statistical inference](https://doi.org/10.1080/01621459.1927.10502953). *Journal of the American Statistical Association*, 22(158), 209–212.
7. Gu, J., Stevens, S., Campolongo, E. G., et al. (2025). [BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](https://doi.org/10.48550/arXiv.2505.23883). The experiment used the [BioCLIP 2.5 Huge model](https://huggingface.co/imageomics/bioclip-2.5-vith14).
8. Siméoni, O., Vo, H. V., Seitzer, M., et al. (2025). [DINOv3](https://doi.org/10.48550/arXiv.2508.10104). arXiv:2508.10104.
9. Bolya, D., Huang, P.-Y., Sun, P., et al. (2025). [Perception Encoder: The best visual embeddings are not at the output of the network](https://arxiv.org/abs/2504.13181). arXiv:2504.13181.

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="../javascripts/image-finetuning-results.js"></script>

<style>
.takeaway-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
  gap: 0.75rem;
  margin: 0.8rem 0 1.4rem;
}
.takeaway-grid article {
  padding: 0.8rem 0.9rem 0.9rem;
  background: var(--md-code-bg-color);
  border-top: 3px solid var(--md-primary-fg-color);
}
.takeaway-grid h3 {
  margin: 0 0 0.35rem;
  font-size: 0.95rem;
}
.takeaway-grid p {
  margin: 0.25rem 0;
  font-size: 0.8rem;
}
.results-downloads {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 1rem 0;
}
.results-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1rem;
  align-items: end;
  margin: 0.75rem 0;
}
.results-controls label {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 11rem;
  font-size: 0.78rem;
}
.results-controls select,
.results-controls input[type="search"] {
  min-height: 2rem;
  padding: 0.3rem 0.45rem;
  color: var(--md-typeset-color);
  background: var(--md-default-bg-color);
  border: 1px solid var(--md-default-fg-color--lighter);
  border-radius: 0.15rem;
}
.results-chart {
  width: 100%;
  min-height: 26rem;
}
.results-caption,
.results-count {
  color: var(--md-default-fg-color--light);
  font-size: 0.78rem;
}
.results-table-wrap {
  width: 100%;
  max-height: 34rem;
  overflow: auto;
  border: 1px solid var(--md-default-fg-color--lightest);
  margin: 0.5rem 0 1rem;
}
.results-table-wrap:focus {
  outline: 2px solid var(--md-typeset-a-color);
  outline-offset: 2px;
}
.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.68rem;
  white-space: nowrap;
}
.results-table caption {
  padding: 0.45rem;
  color: var(--md-default-fg-color--light);
  text-align: left;
  white-space: normal;
}
.results-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--md-default-bg-color);
}
.results-table th button {
  width: 100%;
  padding: 0.45rem;
  color: inherit;
  font: inherit;
  font-weight: 700;
  text-align: left;
  background: transparent;
  border: 0;
  cursor: pointer;
}
.results-table td {
  padding: 0.35rem 0.45rem;
  border-top: 1px solid var(--md-default-fg-color--lightest);
}
.results-table tbody tr:nth-child(even) {
  background: var(--md-code-bg-color);
}
.results-pager {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin: 0.5rem 0;
}
.results-pager .md-button {
  padding: 0.25rem 0.55rem;
}
.results-details {
  margin: 1rem 0 1.5rem;
  padding: 0.7rem 0.9rem;
  border: 1px solid var(--md-default-fg-color--lightest);
}
.results-details summary {
  cursor: pointer;
  font-weight: 700;
}
.results-error {
  color: var(--md-typeset-a-color);
  font-weight: 700;
}
.results-chart-unavailable {
  min-height: auto;
  padding: 1rem;
  color: var(--md-default-fg-color--light);
  background: var(--md-code-bg-color);
}
@media (max-width: 44rem) {
  .results-controls label {
    width: 100%;
  }
  .results-chart {
    min-height: 22rem;
  }
}
</style>
