import numpy as np
from simweave.continuous.systems.full_car import FullCarModel

def compute_full_car_metrics(result, model: FullCarModel | None = None):
    """
    Compute relevant engineering metrics.
    Use optional 'model' input for tyre forces or default to displacement only.
    """
    state = np.asarray(result.state)

    z_s = state[:, 0]
    theta = state[:, 2]
    phi = state[:, 4]

    # velocities
    z_s_dot = state[:, 1]

    # numerical acceleration
    dt = np.mean(np.diff(result.time))
    z_s_ddot = np.gradient(z_s_dot, dt)
    
    # rms of acceleration
    body_accel = z_s_ddot
    rms = np.sqrt(np.mean(body_accel**2))

    # suspension travel
    z_ufl = state[:, 6]
    z_ufr = state[:, 8]
    z_url = state[:, 10]
    z_urr = state[:, 12]

    travel = {
        "fl": z_s - z_ufl,
        "fr": z_s - z_ufr,
        "rl": z_s - z_url,
        "rr": z_s - z_urr,
    }

    
    if model is not None:
        # tyre stiffness
        k_t = model.k_t

        # gather road positions
        z_rfl = result.inputs[:, 0]
        z_rfr = result.inputs[:, 1]
        z_rrl = result.inputs[:, 2]
        z_rrr = result.inputs[:, 3]

        # tyre force loads
        F_fl = k_t * (z_ufl - z_rfl)
        F_fr = k_t * (z_ufr - z_rfr)
        F_rl = k_t * (z_url - z_rrl)
        F_rr = k_t * (z_urr - z_rrr)

        tyre = {
            "fl": F_fl,
            "fr": F_fr,
            "rl": F_rl,
            "rr": F_rr,
        }
        tyre_metric = {
            "name": "Tyre Loads",
            "unit": "N"
        } 
    else:
        # tyre deflection (proxy for load)
        tyre = {
            "fl": z_ufl,
            "fr": z_ufr,
            "rl": z_url,
            "rr": z_urr,
        }   
        tyre_metric = {
            "name": "Tyre deflection",
            "unit": "m"
        } 

    return {
        "body_accel": body_accel,
        "body_accel_RMS": rms,
        "pitch": theta,
        "roll": phi,
        "suspension_travel": travel,
        "tyre": tyre,
        "tyre_metric": tyre_metric
    }