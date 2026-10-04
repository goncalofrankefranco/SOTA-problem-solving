# Color Quantization — Stage I (2024)

**Problem name:** Color Quantization (`kwantyzacja_kolorow`)

**Domain:** Unsupervised image compression, color clustering

**Metric:** Per-image `2 × MSE + 21 × max_color_cost + 42 × mean_color_cost`; lower is better. The output must contain exactly 37 RGB colors. The organizer awards full credit below 8,000 and zero above 8,900.

## Abridged statement

For each 512×512 RGB image, produce a 37-color reconstruction that balances pixelwise squared error against the distance of the chosen colors from the eight RGB cube corners (black, white, red, green, blue, yellow, magenta, cyan). Each image is its own fitting set; no pretrained weights are used.

## Dataset analysis and EDA

The local mirror contains 15 training images and five released validation images. The starter resizes each JPEG to 512×512 and measures the score in 8-bit RGB. Images are evaluated independently, with no labels or paired training targets. The official starter's ordinary 37-cluster K-means baseline reports per-image validation scores of 9,004.565, 8,084.809, 9,634.555, 10,236.788, and 8,945.656, for a mean of 9,181.275.

## Experiments

- **Organizer K-means baseline:** mean validation score 9,181.275; it optimizes MSE but ignores the color-cost terms.
- **Per-image MiniBatch K-means plus the analytic 10.5-unit shrink toward the nearest simple color:** mean 8,441.3 on the released validation images. This reduced the mean color cost but did not reach the full-credit boundary.
- **Same shrink with a palette cost cap of 180:** mean 8,398.8. The maximum-cost term fell slightly, but the mean remained above 8,000.
- **Same shrink with a palette cost cap of 140:** per-image validation scores 8,507.6, 8,063.2, 7,208.9, 6,974.3, and 7,981.7; mean 7,747.1.
- **Same shrink with a palette cost cap of 100:** per-image validation scores **8,121.5, 7,099.6, 6,307.2, 6,101.4, and 7,124.6**; mean **6,950.8**. This is the selected candidate. The 15 training images averaged 6,935.3 with the same settings, versus 7,607.7 for cap 140 and 8,460.2 for cap 180.
- **Stronger 20-unit shrink:** with cap 100 it scored 7,362.8 mean over the 15 training images, worse than the 10.5-unit update; it was not selected.
- A wider-cap exploratory run (radius 200/220) with an earlier slower initialization also averaged about 8,462.5; it was superseded by the faster, repeatable MiniBatch version.

The cluster-centroid update follows the objective: minimizing `2 * squared_error + 42 * color_cost` shifts a mean centroid by `42 / 4 = 10.5` RGB units toward its closest cube corner. A 140-unit cap limits the largest active palette color cost, trading additional MSE for a substantial reduction in the 21-weighted maximum-cost term. Empty palette entries are activated by a single-pixel reassignment so the official 37-color assertion is met.

## Selected solution

Run MiniBatch K-means on 25,000 sampled pixels from each input image, then perform six assignment/centroid updates using the exact weighted per-pixel terms, the analytic 10.5-unit shrink, and the 100-unit palette-cost cap. Return exactly 37 distinct `uint8` RGB colors. Code is in [color_quantization.py](../../../solutions/2024/stage1/color_quantization.py); it reads only the supplied image and does not copy the dataset.

## Validation score and evidence

The best measured released-validation mean is **6,950.8**, below 8,000 (**1.5/1.5 task points, 100/100**). All five per-image scores are listed above. The selected source settings took 22.66 seconds on CPU for all five validation images. No hidden-test result is claimed.

## Compute and runtime

The candidate uses scikit-learn MiniBatch K-means and NumPy/SciPy distance calculations on CPU. The selected implementation took 22.66 seconds for five images, well below the three-minute Google Colab GPU limit. It uses one 25,000-pixel sample for palette initialization and six full-image assignment passes.

## Alternatives

Ordinary K-means is the organizer's MSE-only baseline. The selected constrained codebook improves the full objective. Median-cut, octree quantization, perceptual-space clustering, and stronger/softer palette caps are plausible alternatives but were not measured.

## Progressive hints

1. Compute the ordinary 37-color K-means baseline and inspect each term in the score.
2. Which weighted term can change substantially while only modestly affecting reconstruction error?
3. Move each centroid toward its nearest allowed vivid corner; the weighted objective gives a 10.5-unit proximal shift.
4. Cap the largest palette color cost and verify that all 37 colors are actually used.

**One-line solution:** Use per-image MiniBatch K-means, shrink each centroid toward a simple RGB corner, and cap palette cost at 100.
