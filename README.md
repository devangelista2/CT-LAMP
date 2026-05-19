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

## Notes

- Main config: `configs/base.yaml`
- Mayo override: `configs/mayo_sparse_ct.yaml`
- Outputs are written to `outputs/{timestamp}/`
