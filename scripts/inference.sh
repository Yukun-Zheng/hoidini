#!/bin/bash
# export WANDB_MODE=offline
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
OUT_DIR=hoidini_results


python hoidini/cphoi/cphoi_inference.py \
    --config-name="0_base_config.yaml" \
    model_path=hoidini_training/cphoi_v0/opt000120000.pt \
    out_dir=$OUT_DIR \
    device=7 \
    dno_options_phase1.num_opt_steps=200 \
    dno_options_phase2.num_opt_steps=200 \
    sampler_config.n_samples=1 \
    sampler_config.n_frames=115 \
    render_video=${RENDER_VIDEO:-true} \
    render_resolution=${RENDER_RESOLUTION:-1024} \
    render_fps=${RENDER_FPS:-20} \
    anim_setup=MESH_PARTIAL
    # n_simplify_hands=700 \
    # n_simplify_object=700 \

# Uncomment n_simplify_hands and n_simplify_object to run with low GPU memory (will harm the quality)