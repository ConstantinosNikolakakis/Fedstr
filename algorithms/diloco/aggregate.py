"""
DiLoCo Outer Optimization
=========================
Implements the outer optimization step of DiLoCo (Douillard et al., 2023).
Algorithm 6 from the FEDSTR paper.

Outer optimizer: Nesterov momentum (FedMom, Algorithm 3)

Returns JSON with two fields:
    theta_global: base64-encoded updated global model
    v_t:          base64-encoded momentum state (persisted across rounds)
"""

def aggregate_models(model_params_b64_list, sample_counts,
                     theta_global_b64=None, v_t_b64=None,
                     lr_outer=0.7, momentum=0.9):
    import torch
    import io
    import base64
    import json
    
    # Bootstrap round fallback
    if theta_global_b64 is None:
        print("  Bootstrap round: weighted average (no theta_global yet)")
        total_samples = sum(sample_counts)
        weights = [c / total_samples for c in sample_counts]
        models = []
        for p in model_params_b64_list:
            sd = torch.load(io.BytesIO(base64.b64decode(p)),
                            weights_only=True, map_location="cpu")
            models.append(sd)
        agg = {k: torch.zeros_like(models[0][k]) for k in models[0]}
        for m, w in zip(models, weights):
            for k in m:
                agg[k] += w * m[k]
        buf = io.BytesIO()
        torch.save(agg, buf)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return json.dumps({"theta_global": b64, "v_t": None})

    print(f"  DiLoCo outer step: lr={lr_outer}, momentum={momentum}")

    # Load theta_global
    theta_global = torch.load(
        io.BytesIO(base64.b64decode(theta_global_b64)),
        weights_only=True, map_location="cpu")

    # Load theta_inner from each DVM
    theta_inners = []
    for p in model_params_b64_list:
        sd = torch.load(io.BytesIO(base64.b64decode(p)),
                        weights_only=True, map_location="cpu")
        theta_inners.append(sd)

    print(f"  Computing Delta_outer from {len(theta_inners)} DVMs...")

    # Weighted pseudo-gradient (paper appendix: scale by shard size)
    total_samples = sum(sample_counts)
    weights = [c / total_samples for c in sample_counts]
    delta_outer = {k: torch.zeros_like(theta_global[k])
                   for k in theta_global}
    for ti, w in zip(theta_inners, weights):
        for k in ti:
            delta_outer[k] += w * (theta_global[k] - ti[k])

    # Load or initialize momentum state
    if v_t_b64 is not None:
        print("  Loading momentum state v_{t-1}...")
        v_prev = torch.load(
            io.BytesIO(base64.b64decode(v_t_b64)),
            weights_only=True, map_location="cpu")
    else:
        print("  Initializing v_0 = 0 (first outer round)")
        v_prev = {k: torch.zeros_like(delta_outer[k]) for k in delta_outer}

    # Nesterov update (FedMom Algorithm 3):
    #   v_t = m * v_{t-1} + Delta_outer
    #   theta = theta - lr * (m * v_t + Delta_outer)
    print("  Applying Nesterov momentum...")
    v_t = {}
    new_theta = {}
    for k in theta_global:
        v_t[k] = momentum * v_prev[k] + delta_outer[k]
        new_theta[k] = theta_global[k] - lr_outer * (momentum * v_t[k] + delta_outer[k])

    print("  theta_global updated via Nesterov ✓")

    # Serialize theta_global
    buf_theta = io.BytesIO()
    torch.save(new_theta, buf_theta)
    theta_b64_out = base64.b64encode(buf_theta.getvalue()).decode()

    # Serialize v_t
    buf_vt = io.BytesIO()
    torch.save(v_t, buf_vt)
    vt_b64_out = base64.b64encode(buf_vt.getvalue()).decode()

    print("  Momentum state v_t serialized ✓")

    return json.dumps({"theta_global": theta_b64_out, "v_t": vt_b64_out})