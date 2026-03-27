"""
FedAvg Aggregation
==================
Weighted averaging of PyTorch model parameters.
Implements the outer optimization step of FedAvg (McMahan et al., 2017).
"""


def aggregate_models(model_params_b64_list, sample_counts):
    """
    Weighted averaging of PyTorch model parameters.

    Args:
        model_params_b64_list: List of base64-encoded model state dicts
        sample_counts: List of sample counts for each model

    Returns:
        Base64-encoded aggregated model
    """
    import torch
    import io
    import base64

    # Decode all models
    models = []
    for params_b64 in model_params_b64_list:
        params_bytes = base64.b64decode(params_b64)
        buffer = io.BytesIO(params_bytes)
        state_dict = torch.load(buffer, weights_only=True, map_location='cpu')
        models.append(state_dict)

    # Calculate total samples for weighted averaging
    total_samples = sum(sample_counts)
    weights = [count / total_samples for count in sample_counts]

    # Initialize aggregated state dict with zeros
    aggregated = {}
    for key in models[0].keys():
        aggregated[key] = torch.zeros_like(models[0][key])

    # Weighted averaging
    for model, weight in zip(models, weights):
        for key in model.keys():
            aggregated[key] += weight * model[key]

    # Serialize aggregated model
    buffer = io.BytesIO()
    torch.save(aggregated, buffer)
    aggregated_bytes = buffer.getvalue()

    import json
    return json.dumps({'theta_global': base64.b64encode(aggregated_bytes).decode('utf-8'), 'v_t': None})
