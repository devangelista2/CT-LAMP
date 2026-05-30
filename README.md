# CT-LAMP

CT-LAMP is a sparse-view CT reconstruction project based on DDNM+ with multistep DPM-Solver++ style updates.

This codebase is now CT-only:
- Dataset: Mayo (`../data/Mayo`)
- Forward model: sparse-view parallel-beam CT via `IPPy.operators.CTProjector` (ASTRA)
- Methods: `fbp`, `sirt`, `ddnm_plus`, `ct_lamp` (2M negative correction), `ct_lamp_3m`, `score_sde`, `mcg`, `dps`, `ps_plus`, `diffpir`
- Diffusion backend: RD-DGP MONAI Generative UNet by default

The diffusion model operates on images in `[-1, 1]`, while the CT operator acts on
physical images in `[0, 1]`. CT-LAMP now performs data consistency in physical
space and maps the corrected estimate back to diffusion space for denoising updates.
The added `DPS` and `DiffPIR` baselines are adapted the same way, so all
data terms and range projections operate in CT physical units.

## Setup

```bash
uv sync
```

If ASTRA is missing, install it in your runtime environment (for example with conda).

`operator.noise_sigma` is expressed in raw sinogram units, not normalized image units.

Default CT-LAMP parameters are tuned for this raw CT scaling, where `K(x)` can be on
the order of `10^2` and `K^T K(x)` on the order of `10^3`.

## Pretrained RD-DGP weights

Default config is already set to RD-DGP MONAI weights:

- `model.checkpoint_path=ct_lamp/checkpoints/rd_dgp_monai_finetuned_best.pth`
- `model.model_config_path=configs/rd_dgp_monai_unet.yaml`

These values can be overridden from CLI.

The loader uses `generative.networks.nets.DiffusionModelUNet` (same class as RD-DGP) when available.

## Run

```bash
uv run python scripts/run.py \
  --config configs/mayo_sparse_ct.yaml \
  --method ct_lamp \
  data.num_images=1
```

## Compare methods

```bash
  uv run python scripts/compare.py \
  --config configs/mayo_sparse_ct.yaml \
  --methods fbp,sirt,ddnm_plus,ct_lamp,ct_lamp_3m,score_sde,mcg,dps,ps_plus,diffpir \
  data.num_images=1
```

## Generate prior samples grid

```bash
uv run python scripts/generate_grid.py \
  --config configs/mayo_sparse_ct.yaml \
  --num_images 16 \
  --num_steps 100 \
  --sampler ddim \
  --output outputs/sample_grid.png
```

## CT-LAMP beta schedule expression

`ct_lamp` supports two equivalent lag parameterizations:

- `ct_lamp.gamma`: a fixed lag strength
- `ct_lamp.beta_schedule`: a direct specification of `beta_t`

If `beta_schedule` is set, it overrides `gamma`.

The most flexible option is:

```yaml
ct_lamp:
  beta_schedule:
    kind: "expression"
    expression: "0.03"
```

The `expression` string is evaluated at every reverse step and must return a single numeric value,
which is used as the current `beta_t` in

```text
D_tilde = (1 - beta_t) * D_cur + beta_t * D_prev
```

The following variables are available inside the expression:

- `t`, `t_cur`: current reverse timestep
- `t_prev`: next reverse timestep after the update
- `step_idx`: zero-based reverse-step index
- `num_steps`: total configured number of reverse steps
- `frac`: normalized step position in `[0, 1]`
- `h`: current log-SNR gap `lambda(t_prev) - lambda(t_cur)`
- `h_prev`: previous log-SNR gap
- `lambda_cur`, `lambda_prev`: current and next log-SNR values
- `gamma`: the configured `ct_lamp.gamma` value

The following functions/constants are available:

- `abs`, `min`, `max`
- `sqrt`, `exp`, `log`
- `sin`, `cos`, `tan`
- `pi`

Examples:

```yaml
ct_lamp:
  beta_schedule:
    kind: "expression"
    expression: "0.03"
```

Constant lagged averaging.

```yaml
ct_lamp:
  beta_schedule:
    kind: "expression"
    expression: "0.01 + 0.04 * frac"
```

Linearly increases `beta_t` during sampling.

```yaml
ct_lamp:
  beta_schedule:
    kind: "expression"
    expression: "min(0.05, 0.5 * h / max(h_prev, 1e-8))"
```

Makes `beta_t` depend on the ratio between consecutive log-SNR gaps.

```yaml
ct_lamp:
  beta_schedule:
    kind: "expression"
    expression: "0.03 * (1 + cos(pi * frac)) / 2"
```

Starts larger and decays smoothly toward zero.

Use CLI overrides as usual, for example:

```bash
uv run python scripts/run.py \
  --config configs/mayo_sparse_ct.yaml \
  --method ct_lamp \
  'ct_lamp.beta_schedule.kind=expression' \
  'ct_lamp.beta_schedule.expression=min(0.05, 0.01 + 0.04 * frac)'
```

## Notes

- Main config: `configs/base.yaml`
- Mayo override: `configs/mayo_sparse_ct.yaml`
- Outputs are written to `outputs/{timestamp}/`
