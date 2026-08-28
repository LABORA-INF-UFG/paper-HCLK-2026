import math
import numpy as np
import tensorflow as tf

def compute_steering_vector(
    tx_positions,
    rx_positions,
    tx_array,
):
    """
    Args:
        tx_positions: List of TX sector positions (e.g., 3 sectors) as [x, y, z] arrays. Shape: [num_tx, 3].
        rx_positions: UE positions as [x, y, z] arrays. Shape: [num_rx, 3].

    Returns:
        tx_vectors: Steering vectors with shape [num_rx, 1, num_tx, num_tx_ant, 1, 1].
    """
    num_rx = len(rx_positions)
    num_tx = len(tx_positions)
    num_tx_ant = tx_array.num_ant

    # Initialize output tensor
    tx_vectors = np.zeros((num_rx, 1, num_tx, num_tx_ant, 1, 1), dtype=np.complex64)

    for rx_idx in range(num_rx):
        for tx_idx in range(num_tx):
            # Compute direction from TX to RX
            delta = rx_positions[rx_idx] - tx_positions[tx_idx]
            phi = np.arctan2(delta[1], delta[0])  # Azimuth
            theta = np.arctan2(np.linalg.norm(delta[:2]), delta[2])  # Elevation

            # Compute steering vector
            pos = np.array(tx_array.normalized_positions)  # Shape: (3, num_tx_ant)
            k = np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )
            phases = (
                2 * np.pi * (pos[0, :] * k[0] + pos[1, :] * k[1] + pos[2, :] * k[2])
            )
            steering_vec = np.exp(-1j * phases)  # Shape: (num_tx_ant,)

            # Assign to output
            tx_vectors[rx_idx, 0, tx_idx, :, 0, 0] = steering_vec

    return tf.convert_to_tensor(tx_vectors, dtype=tf.complex64)


def apply_precoder(cir_values, tx_vectors):
    """
    Args:
        cir_values: CIR with shape [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths].
        tx_vectors: Steering vectors with shape [num_rx, 1, num_tx, num_tx_ant, 1].

    Returns:
        effective_cir: Beamformed CIR with shape [num_rx, num_rx_ant, num_tx, 1, num_paths].
    """
    # Ensure inputs are complex
    tx_vectors = tf.cast(tx_vectors, dtype=tf.complex64)

    # Broadcast tx_vectors to match CIR shape
    tx_vectors = tf.broadcast_to(tx_vectors, tf.shape(cir_values))

    # Apply conjugate beamforming and sum over TX antennas
    effective_cir = tf.reduce_sum(
        cir_values * tf.math.conj(tx_vectors), axis=3, keepdims=True
    )

    return effective_cir


def calc_thr_per_antenna(
    cir_values,
    tx_power_dBm,
    noise_power_DBM,
    bandwidth_Hz,
    is_beamformed=True,
    worstcase=True,  # TODO
):
    """
    Calculate throughput with MRC for non-beamformed cases.

    Args:
        cir_values: Shape for beamformed: [rx, rx_ant, tx, tx_ant, num_paths, num_time_steps]
        tx_power_dBm: Total transmit power in dBm
        noise_power_DBM: Noise power in dBm
        bandwidth_Hz: System bandwidth
        is_beamformed: Whether TX uses beamforming

    Returns:
        Throughput in Mbps [batch, rx, tx]
    """
    # Power conversions
    tx_power_W = 1e-3 * 10 ** (tx_power_dBm / 10)
    noise_power_W = 1e-3 * 10 ** (noise_power_DBM / 10)

    if is_beamformed:
        # Beamformed case - power already focused
        channel_power = tf.reduce_mean(tf.abs(cir_values) ** 2, axis=[1, 3, 4, 5])
        received_power = tx_power_W * channel_power
    else:
        # Non-beamformed case: equal power allocation
        tx_power_per_component = tx_power_W / cir_values.shape[-3]

        # Equal power allocation (average over RX and TX antennas)
        channel_power = tf.reduce_mean(
            tf.abs(cir_values) ** 2, axis=[1, 3, 4, 5]
        )  # Avg over rx_ant and tx_ant
        received_power = tx_power_per_component * channel_power

    # SNR and capacity calculation
    if worstcase:
        interference_power = (
            tf.reduce_sum(received_power, axis=1, keepdims=True) - received_power
        )
        sinr = received_power / (noise_power_W + interference_power)
    else:
        sinr = received_power / noise_power_W
    capacity = tf.math.log(1 + sinr) / tf.math.log(2.0)
    thr_mbps = capacity * bandwidth_Hz / 1e6

    return tf.squeeze(thr_mbps)

def transform_xy_sumo_to_sionna(x, y, cx, cy, map_rot_deg, offset_xy):
    """Converts SUMO (x, y) coordinates to the system used by Sionna:

    1) recenter: corner -> center
    2) rotates in the plane by 'map_rot_deg'
    3) applies a global offset 'offset_xy' (dx, dy)
    Accepts scalars or NumPy arrays of the same shape. Returns (xr, yr).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1) recenter
    x0 = x - cx
    y0 = y - cy

    # 2) rotate
    th = np.deg2rad(map_rot_deg)
    c, s = np.cos(th), np.sin(th)
    xr = c * x0 - s * y0
    yr = s * x0 + c * y0

    # 3) global offset
    dx, dy = float(offset_xy[0]), float(offset_xy[1])
    xr = xr + dx
    yr = yr + dy

    return xr, yr