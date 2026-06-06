import numpy as np
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


def generate_dst_basis(grid_size, max_frequency):
    """
    Returns basis array of shape (n_basis, N, N).
    Keeps all (k,l) with k <= max_frequency and l <= max_frequency (square cutoff).
    k, l start at 1 (no DC component).
    """
    basis = []
    x = np.arange(grid_size)

    for k in range(1, max_frequency + 1):
        for l in range(1, max_frequency + 1):
            row = np.sqrt(2 / grid_size) * np.sin(np.pi * k * (x + 1) / (grid_size + 1))
            col = np.sqrt(2 / grid_size) * np.sin(np.pi * l * (x + 1) / (grid_size + 1))
            basis.append(np.outer(row, col))

    return np.array(basis)


def create_physical_setup(
    grid_size, space_step_size, sound_speed, detector_distance_from_boundary, pml_width
):

    N = Vector([grid_size, grid_size])
    d = Vector([space_step_size, space_step_size])
    kgrid = kWaveGrid(N, d)

    medium = kWaveMedium(sound_speed=sound_speed)

    sensor_mask = np.zeros((grid_size, grid_size))
    sensor_mask[
        detector_distance_from_boundary : (grid_size - detector_distance_from_boundary),
        detector_distance_from_boundary,
    ] = 1
    sensor_mask[
        (grid_size - detector_distance_from_boundary),
        detector_distance_from_boundary : (grid_size - detector_distance_from_boundary),
    ] = 1
    sensor = kSensor(mask=sensor_mask)

    kgrid.makeTime(medium.sound_speed)

    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False, show_sim_log=False
    )
    simulation_options = SimulationOptions(
        save_to_disk=True, data_cast="single", pml_size=pml_width, use_kspace=True
    )

    return kgrid, sensor, medium, execution_options, simulation_options


def assemble_waves_from_dst_functions(
    dst_functions,
    grid_size,
    pic_size,
    space_step_size,
    sound_speed,
    detector_distance_from_boundary,
    pml_width,
):

    measurements_list = []
    for grid_vals in dst_functions:
        p0 = np.pad(
            grid_vals,
            (
                ((grid_size - pic_size) // 2, (grid_size - pic_size) // 2),
                ((grid_size - pic_size) // 2, (grid_size - pic_size) // 2),
            ),
            mode="constant",
            constant_values=0,
        )

        kgrid, sensor, medium, execution_options, simulation_options = (
            create_physical_setup(
                grid_size,
                space_step_size,
                sound_speed,
                detector_distance_from_boundary,
                pml_width,
            )
        )
        source = kSource()
        source.p0 = p0
        sim = kspaceFirstOrder2D(
            kgrid, source, sensor, medium, simulation_options, execution_options
        )
        measurements = sim["p"].copy()
        measurements_list.append(measurements.ravel())

    return np.vstack(measurements_list)


def dst_representation(function, dst_functions, space_step_size):
    matrix = (
        space_step_size**2 * np.vstack([shape.ravel() for shape in dst_functions]).T
    )
    function_vector = space_step_size**2 * function.ravel()

    lstsq_representation = np.linalg.solve(
        matrix.T @ matrix, matrix.T @ function_vector
    )

    return np.sum(
        [lstsq_representation[i] * dst_functions[i] for i in range(len(dst_functions))],
        axis=0,
    )


def H_1_0_product(function_1, function_2, space_step_size):

    grad_1 = np.gradient(function_1)
    grad_2 = np.gradient(function_2)

    return np.sum(
        [
            np.vdot(grad_1[i], grad_2[i]) * (space_step_size**2)
            for i in range(len(grad_1))
        ]
    )


def L_2_product(function_1, function_2, space_step_size):

    return np.vdot(function_1, function_2) * space_step_size**2


def L_2_measurement_surface_product(
    measurements_1, measurements_2, space_step_size, time_step_size
):

    return np.vdot(measurements_1, measurements_2) * space_step_size * time_step_size


def forward(
    f,
    grid_size,
    space_step_size,
    sound_speed,
    detector_distance_from_boundary,
    pml_width,
):
    kgrid, sensor, medium, execution_options, simulation_options = (
        create_physical_setup(
            grid_size,
            space_step_size,
            sound_speed,
            detector_distance_from_boundary,
            pml_width,
        )
    )
    source = kSource()
    source.p0 = f

    sim = kspaceFirstOrder2D(
        kgrid, source, sensor, medium, simulation_options, execution_options
    )
    measurements = sim["p"].copy()

    return measurements


def time_reversal(
    measurements,
    grid_size,
    pic_size,
    sound_speed,
    detector_distance_from_boundary,
    pml_width,
):
    N = Vector([grid_size, grid_size])
    d = Vector([grid_size, grid_size])
    kgrid = kWaveGrid(N, d)
    medium = kWaveMedium(sound_speed=sound_speed)

    p_mask = np.zeros((grid_size, grid_size))
    p_mask[
        detector_distance_from_boundary : (grid_size - detector_distance_from_boundary),
        detector_distance_from_boundary,
    ] = 1
    p_mask[
        (grid_size - detector_distance_from_boundary),
        detector_distance_from_boundary : (grid_size - detector_distance_from_boundary),
    ] = 1

    final_time_step = measurements.shape[0]
    measurements = measurements.T
    source = kSource()
    source.p_mask = p_mask
    source.p = np.flip(measurements, axis=-1).copy()

    sensor = kSensor(mask=np.ones((grid_size, grid_size)))
    kgrid.makeTime(medium.sound_speed)
    execution_options = SimulationExecutionOptions(is_gpu_simulation=False)
    simulation_options = SimulationOptions(
        save_to_disk=True, data_cast="single", pml_size=pml_width, use_kspace=True
    )
    rec_sim = kspaceFirstOrder2D(
        kgrid, source, sensor, medium, simulation_options, execution_options
    )

    p_rec = rec_sim["p"].T[..., final_time_step - 1].copy()
    p_rec = p_rec.reshape(grid_size, grid_size)

    pad_size = (grid_size - pic_size) // 2
    p_rec = p_rec[pad_size : (grid_size - pad_size), pad_size : (grid_size - pad_size)]
    p_rec = np.pad(
        p_rec, ((pad_size, pad_size), (pad_size, pad_size)), constant_values=0
    )

    return p_rec.T


def L_2_error(true, approx):
    diff = true - approx
    true_norm = np.sqrt(L_2_product(true, true))
    abs_error = np.sqrt(L_2_product(diff, diff))
    rel_error = abs_error / true_norm

    print(f"Absolute error: {abs_error}")
    print(f"Relative error: {rel_error}")

    return rel_error


def disk_phantom(pic_size):
    x = np.linspace(-1, 1, pic_size)
    y = np.linspace(-1, 1, pic_size)
    X, Y = np.meshgrid(x, y)

    phantom = np.zeros_like(X)
    r = 0.08
    diagonal_disks = [
        (-0.6, 0.6),  # near L-corner
        (-0.3, 0.3),
        (0.0, 0.0),  # center
        (0.3, -0.3),
        (0.6, -0.6),  # far corner — disappears first
    ]
    off_diagonal_disks = [
        (0.6, 0.4),  # far from left detector, close to bottom
        (-0.4, -0.6),  # close to left detector, far from bottom
        (0.3, 0.3),  # far from both but not on diagonal
        (-0.2, -0.2),  # close to both but not on diagonal
    ]
    for cx, cy in diagonal_disks:
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r**2
        phantom[mask] = 1.0
    for cx, cy in off_diagonal_disks:
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r**2
        phantom[mask] = 0.6

    return phantom


def tikhonov_reconstruction(
    measurements,
    alpha,
    grid_size,
    pic_size,
    max_frequency,
    space_step_size,
    time_step_size,
    sound_speed,
    detector_distance_from_boundary,
    pml_width,
):

    dst_functions = generate_dst_basis(pic_size, max_frequency)
    dst_waves = assemble_waves_from_dst_functions(
        dst_functions,
        grid_size,
        pic_size,
        space_step_size,
        sound_speed,
        detector_distance_from_boundary,
        pml_width,
    )
    lhs_matrix = dst_waves @ dst_waves.T
    stiffness_matrix = np.diag(
        [
            (k**2 + l**2) * np.pi**2
            for k in range(1, max_frequency + 1)
            for l in range(1, max_frequency + 1)
        ]
    )
    measurements = measurements.ravel()
    rhs_vector = dst_waves @ measurements

    lhs_matrix = space_step_size * time_step_size * lhs_matrix
    rhs_vector = space_step_size * time_step_size * rhs_vector
    lhs_matrix = lhs_matrix + alpha * stiffness_matrix

    x_coefficients = np.linalg.solve(lhs_matrix, rhs_vector)

    x_reconstructed = np.sum(
        [x_coefficients[i] * dst_functions[i] for i in range(len(dst_functions))], 0
    )
    x_reconstructed = np.pad(
        x_reconstructed,
        (
            ((grid_size - pic_size) // 2, (grid_size - pic_size) // 2),
            ((grid_size - pic_size) // 2, (grid_size - pic_size) // 2),
        ),
        mode="constant",
        constant_values=0,
    )

    return x_reconstructed


def tikhonov_reconstruction_reduced_time(
    measurements,
    alpha,
    grid_size,
    pic_size,
    max_frequency,
    space_step_size,
    time_step_size,
    sound_speed,
    detector_distance_from_boundary,
    pml_width,
    max_time_steps,
):

    dst_functions = generate_dst_basis(pic_size, max_frequency)
    dst_waves = assemble_waves_from_dst_functions(
        dst_functions,
        grid_size,
        pic_size,
        space_step_size,
        sound_speed,
        detector_distance_from_boundary,
        pml_width,
    )
    reduced_dst_waves_list = []
    n_detector_points = dst_waves[0].shape[0] / max_time_steps
    final_time_step = measurements.shape[0]
    for j in range(dst_waves.shape[0]):
        reduced_dst_waves_list.append(
            dst_waves[j]
            .reshape((max_time_steps, n_detector_points))[:final_time_step, :]
            .ravel()
        )
    reduced_dst_waves = np.vstack(reduced_dst_waves_list)

    lhs_matrix = reduced_dst_waves @ reduced_dst_waves.T
    stiffness_matrix = np.diag(
        [
            (k**2 + l**2) * np.pi**2
            for k in range(1, max_frequency + 1)
            for l in range(1, max_frequency + 1)
        ]
    )
    measurements = measurements.ravel()
    rhs_vector = reduced_dst_waves @ measurements

    lhs_matrix = space_step_size * time_step_size * lhs_matrix
    rhs_vector = space_step_size * time_step_size * rhs_vector
    lhs_matrix = lhs_matrix + alpha * stiffness_matrix

    x_coefficients = np.linalg.solve(lhs_matrix, rhs_vector)

    x_reconstructed = np.sum(
        [x_coefficients[i] * dst_functions[i] for i in range(len(dst_functions))], 0
    )
    x_reconstructed = np.pad(
        x_reconstructed,
        (
            ((grid_size - pic_size) // 2, (grid_size - pic_size) // 2),
            ((grid_size - pic_size) // 2, (grid_size - pic_size) // 2),
        ),
        mode="constant",
        constant_values=0,
    )

    return x_reconstructed
