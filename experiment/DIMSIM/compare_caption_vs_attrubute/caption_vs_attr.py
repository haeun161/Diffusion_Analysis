"""
Concept vs Caption vs Attribute 비교 실험  (단일 SEED)
=====================================================

목적
----
한 시드에서
  (a) bare concept prompt   (예: "airplane")
  (b) BLIP caption prompt
  (c) simple_attribute_prompts 의 각 attribute prompt
세 종류 prompt 로 SD3 를 돌려 비교한다.

각 JointTransformerBlock 내부 분해 (sd3_hook 와 동일):
    L_output = L_minus_1 + delta_attn + delta_mlp
                  ↑           ↑              ↑
                 lm1          da             dm

흐름
----
컨셉마다
  concept(seed) 1회 생성 → concept_stats
  caption(seed) 1회 생성 → cap_stats
  attribute 각각에 대해 (seed) 생성 → attr_stats
    └ 3쌍의 cosine sim:
        concept-vs-caption  (attribute 무관, 컨셉당 1번 계산)
        concept-vs-attr
        caption-vs-attr
    └ 3행 × 3열 heatmap PNG 저장
    └ concept-vs-attr / caption-vs-attr 누적
  → attribute 평균 3행 × 3열 heatmap PNG 저장
    (concept-vs-caption 행은 평균 대상 아님 → 그대로 표기)

저장물
------
output/caption_vs_attr/
├── {concept}/
│   ├── images/concept_seed{S}.png
│   ├── images/caption_seed{S}.png
│   ├── images/{at}_{av}_seed{S}.png
│   └── per_attr/{at}_{av}.png            # 3행 × 3열
└── {concept}_caption_vs_attr.png         # 3행 × 3열 (attribute 평균)

실행
----
  python caption_vs_attr.py
"""

import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
DEVICE = "cuda"
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
NUM_STEPS = 28
GUIDANCE_SCALE = 7.0
HIDDEN_DIM = 1536

CONCEPTS = ["airplane", "bird", "car", "clock",
            "couch", "elephant", "train", "umbrella"]
SEED = 3
COMPONENTS = ["lm1", "da", "dm"]   # L_minus_1, delta_attn, delta_mlp
COMPONENT_LABELS = {
    "lm1": "L_minus_1  (block input)",
    "da":  "delta_attn  (attention)",
    "dm":  "delta_mlp  (MLP)",
}

ROOT = Path(__file__).resolve().parent
CAPTIONS_PATH = ROOT / "blip_captions.json"
ATTR_DIR = Path("/home/haeun/Diffusion_Analysis/DIMCIM/COCO-DIMCIM/simple_attribute_prompts")
OUT_DIR = ROOT / "output" / "caption_vs_attr"
SD3_HOOK_DIR = ROOT.parent.parent / "mode_path"


# ────────────────────────────────────────────────────────────
# Prompt 로딩 (seed 0 한 개만 사용)
# ────────────────────────────────────────────────────────────
def load_prompts():
    with open(CAPTIONS_PATH) as f:
        captions_all = json.load(f)
    captions = {c: captions_all[c][str(SEED)] for c in CONCEPTS}

    attr_data = {}
    for c in CONCEPTS:
        with open(ATTR_DIR / f"{c}_simple_attribute_prompts.json") as f:
            d = json.load(f)
        flat = [
            (at, av, p)
            for at, vs in d["simple_attribute_prompts"].items()
            for av, p in vs.items()
        ]
        attr_data[c] = flat
    return captions, attr_data


# ────────────────────────────────────────────────────────────
# Pipeline + hook
# ────────────────────────────────────────────────────────────
def build_pipeline_with_hook():
    sys.path.insert(0, str(SD3_HOOK_DIR))
    from sd3_hook import attach_step_tracker, patch_transformer
    from diffusers import StableDiffusion3Pipeline

    print("Loading SD3 pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    NUM_BLOCKS = len(pipe.transformer.transformer_blocks)
    print(f"✓ {NUM_BLOCKS}개 JointTransformerBlock")

    # 한 run 내부에서만 살아있는 메모리 버퍼
    buf = {
        f"{c}_chan": np.zeros((NUM_STEPS, NUM_BLOCKS, HIDDEN_DIM), dtype=np.float32)
        for c in COMPONENTS
    }
    active = {"on": False}

    def on_capture(block_idx, L_minus_1, L_after_attn, L_output, batch_size):
        if not active["on"]:
            return
        s = step_counter[0]
        if s >= NUM_STEPS:
            return

        # cond branch (CFG batch=2 중 index 1) 만, float32 변환 후 뺄셈
        lm1 = L_minus_1[1].float()      # (4096, 1536)
        lat = L_after_attn[1].float()
        lo  = L_output[1].float()

        da = lat - lm1
        dm = lo  - lat

        for name, t in [("lm1", lm1), ("da", da), ("dm", dm)]:
            buf[f"{name}_chan"][s, block_idx] = t.mean(dim=0).cpu().numpy()

    already = patch_transformer(pipe.transformer, on_capture)
    if already > 0:
        print(f"⚠ {already} 블록 이미 패치됨")
    step_counter = attach_step_tracker(pipe.transformer, on_step=lambda _s: True)
    if step_counter is None:
        raise RuntimeError("step tracker already attached. Restart Python.")

    def run_one(prompt_text: str, seed: int, png_path: Path) -> dict:
        for v in buf.values():
            v.fill(0)
        step_counter[0] = 0
        active["on"] = True

        gen = torch.Generator(device=pipe.device).manual_seed(seed)
        image = pipe(
            prompt_text,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=gen,
        ).images[0]

        active["on"] = False
        png_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(png_path)

        snapshot = {k: v.copy() for k, v in buf.items()}

        del image
        gc.collect()
        torch.cuda.empty_cache()
        return snapshot

    return run_one, NUM_BLOCKS


# ────────────────────────────────────────────────────────────
# cosine sim (step, block) over channel-mean vectors
# ────────────────────────────────────────────────────────────
def cos_sim_per_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a, b : (NUM_STEPS, NUM_BLOCKS, D)
    return: (NUM_STEPS, NUM_BLOCKS)
    """
    num = (a * b).sum(-1)
    da = np.linalg.norm(a, axis=-1)
    db = np.linalg.norm(b, axis=-1)
    return num / (np.maximum(da, 1e-8) * np.maximum(db, 1e-8))


# ────────────────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────────────────
def plot_components(rows: list, title: str, out_png: Path):
    """
    rows : list[ (row_label, {"lm1": (T,B), "da": (T,B), "dm": (T,B)}) ]
    n_rows × 3 cols heatmap. 모든 셀에서 vmin/vmax 공유.
    """
    n_rows = len(rows)
    all_vals = np.concatenate([
        sims[c].ravel()
        for _, sims in rows
        for c in COMPONENTS
    ])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for i, (row_label, sims) in enumerate(rows):
        for j, comp in enumerate(COMPONENTS):
            ax = axes[i, j]
            im = ax.imshow(
                sims[comp], aspect="auto", cmap="RdYlBu_r",
                origin="lower", vmin=vmin, vmax=vmax,
            )
            ax.set_xlabel("block")
            ax.set_ylabel("step")
            if i == 0:
                ax.set_title(COMPONENT_LABELS[comp], fontsize=10, fontweight="bold")
            if j == 0:
                ax.text(
                    -0.32, 0.5, row_label,
                    transform=ax.transAxes,
                    ha="right", va="center",
                    fontsize=10, fontweight="bold",
                )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title}  (vmin={vmin:.3f} vmax={vmax:.3f})", fontsize=12)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captions, attr_data = load_prompts()
    run_one, _NUM_BLOCKS = build_pipeline_with_hook()

    def sims_dict(stats_a: dict, stats_b: dict) -> dict:
        return {
            c: cos_sim_per_block(stats_a[f"{c}_chan"], stats_b[f"{c}_chan"])
            for c in COMPONENTS
        }

    for concept in CONCEPTS:
        cap_prompt = captions[concept]
        attrs = attr_data[concept]

        cdir = OUT_DIR / concept
        img_dir = cdir / "images"
        per_attr_dir = cdir / "per_attr"
        img_dir.mkdir(parents=True, exist_ok=True)
        per_attr_dir.mkdir(parents=True, exist_ok=True)

        n_runs = 2 + len(attrs)   # concept + caption + attrs
        pbar = tqdm(total=n_runs, desc=f"[{concept}]")

        # 1) bare concept
        concept_stats = run_one(
            concept, SEED,
            img_dir / f"concept_seed{SEED}.png",
        )
        pbar.update(1); pbar.set_postfix_str("concept")

        # 2) caption
        cap_stats = run_one(
            cap_prompt, SEED,
            img_dir / f"caption_seed{SEED}.png",
        )
        pbar.update(1); pbar.set_postfix_str("caption")

        # 3) concept vs caption  (attribute 무관, 컨셉당 1번)
        cvc = sims_dict(concept_stats, cap_stats)

        # 4) attribute 별 (concept-vs-caption / concept-vs-attr / caption-vs-attr) 3×3 + 누적
        cva_sum   = {c: np.zeros((NUM_STEPS, _NUM_BLOCKS), dtype=np.float64)
                     for c in COMPONENTS}
        cap_a_sum = {c: np.zeros((NUM_STEPS, _NUM_BLOCKS), dtype=np.float64)
                     for c in COMPONENTS}
        n_attr = 0

        for at, av, ap in attrs:
            attr_stats = run_one(
                ap, SEED,
                img_dir / f"{at}_{av}_seed{SEED}.png",
            )

            cva   = sims_dict(concept_stats, attr_stats)
            cap_a = sims_dict(cap_stats,     attr_stats)

            plot_components(
                [
                    ("concept vs caption", cvc),
                    ("concept vs attr",    cva),
                    ("caption vs attr",    cap_a),
                ],
                title=f'"{concept}" — {at}/{av}  (seed={SEED})\n'
                      f'caption: "{cap_prompt}"\nattribute: "{ap}"',
                out_png=per_attr_dir / f"{at}_{av}.png",
            )

            for c in COMPONENTS:
                cva_sum[c]   += cva[c]
                cap_a_sum[c] += cap_a[c]
            n_attr += 1

            pbar.update(1); pbar.set_postfix_str(f"{at}/{av}")
            del attr_stats

        pbar.close()
        del concept_stats, cap_stats

        if n_attr == 0:
            print(f"[skip] no attribute for {concept}")
            continue

        # 5) attribute 평균 (3×3) — concept-vs-caption 은 평균 대상 아니라 그대로
        cva_mean   = {c: (cva_sum[c]   / n_attr).astype(np.float32)
                      for c in COMPONENTS}
        cap_a_mean = {c: (cap_a_sum[c] / n_attr).astype(np.float32)
                      for c in COMPONENTS}
        out_png = OUT_DIR / f"{concept}_caption_vs_attr.png"
        plot_components(
            [
                ("concept vs caption",       cvc),
                ("concept vs attr (mean)",   cva_mean),
                ("caption vs attr (mean)",   cap_a_mean),
            ],
            title=f'"{concept}" — vs attribute (mean of {n_attr} attrs)  seed={SEED}',
            out_png=out_png,
        )
        print(f"saved: {out_png}")

    print(f"\n✓ 완료. 출력: {OUT_DIR}")


if __name__ == "__main__":
    main()
