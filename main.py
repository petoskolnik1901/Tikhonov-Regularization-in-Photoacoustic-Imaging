import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.data import shepp_logan_phantom
import utils

PIC_SIZE = 128
GRID_SIZE = 256
MAX_FREQUENCY = 70
SPACE_STEP_SIZE = 0.001
TIME_STEP_SIZE = 0.0003
SOUND_SPEED = 1
DETECTOR_DISTANCE = 25
PML_WIDTH = 15
MAX_TIME_STEP = 1207


def method_comparison_shepp():
    dst_functions = utils.generate_dst_basis(PIC_SIZE, MAX_FREQUENCY)
    shepp = shepp_logan_phantom()
    shepp = resize(shepp, (PIC_SIZE, PIC_SIZE), anti_aliasing=True)
    shepp_DST = utils.dst_representation(shepp, dst_functions, SPACE_STEP_SIZE)
    shepp = np.pad(
        shepp,
        (
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
        ),
    )
    shepp_DST = np.pad(
        shepp_DST,
        (
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
        ),
    )

    shepp_measurements = utils.forward(
        shepp, GRID_SIZE, SPACE_STEP_SIZE, SOUND_SPEED, DETECTOR_DISTANCE, PML_WIDTH
    )

    tikhonov_shepp = utils.tikhonov_reconstruction(
        shepp_measurements,
        1.0e-14,
        GRID_SIZE,
        PIC_SIZE,
        MAX_FREQUENCY,
        SPACE_STEP_SIZE,
        TIME_STEP_SIZE,
        SOUND_SPEED,
        DETECTOR_DISTANCE,
        PML_WIDTH,
    )
    reversal_shepp = utils.time_reversal(
        shepp_measurements,
        GRID_SIZE,
        PIC_SIZE,
        SOUND_SPEED,
        DETECTOR_DISTANCE,
        PML_WIDTH,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    images = [shepp, shepp_DST, tikhonov_shepp, reversal_shepp]

    for ax, img in zip(axes.flat, images):
        im = ax.imshow(img, cmap="Greys")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def method_comparison_disks():
    dst_functions = utils.generate_dst_basis(PIC_SIZE, MAX_FREQUENCY)
    disks = utils.disk_phantom(PIC_SIZE)
    disks_DST = utils.dst_representation(disks, dst_functions, SPACE_STEP_SIZE)
    disks = np.pad(
        disks,
        (
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
        ),
    )
    disks_DST = np.pad(
        disks_DST,
        (
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
        ),
    )

    disks_measurements = utils.forward(
        disks, GRID_SIZE, SPACE_STEP_SIZE, SOUND_SPEED, DETECTOR_DISTANCE, PML_WIDTH
    )

    tikhonov_disks = utils.tikhonov_reconstruction(
        disks_measurements,
        1.0e-14,
        GRID_SIZE,
        PIC_SIZE,
        MAX_FREQUENCY,
        SPACE_STEP_SIZE,
        TIME_STEP_SIZE,
        SOUND_SPEED,
        DETECTOR_DISTANCE,
        PML_WIDTH,
    )
    reversal_disks = utils.time_reversal(
        disks_measurements,
        GRID_SIZE,
        PIC_SIZE,
        SOUND_SPEED,
        DETECTOR_DISTANCE,
        PML_WIDTH,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    images = [disks, disks_DST, tikhonov_disks, reversal_disks]

    for ax, img in zip(axes.flat, images):
        im = ax.imshow(img, cmap="Greys")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def method_comparison_reduced_time_shepp(final_time_step=300):
    dst_functions = utils.generate_dst_basis(PIC_SIZE, MAX_FREQUENCY)
    shepp = shepp_logan_phantom()
    shepp = resize(shepp, (PIC_SIZE, PIC_SIZE), anti_aliasing=True)
    shepp_DST = utils.dst_representation(shepp, dst_functions, SPACE_STEP_SIZE)
    shepp = np.pad(
        shepp,
        (
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
        ),
    )
    shepp_DST = np.pad(
        shepp_DST,
        (
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
        ),
    )

    shepp_measurements = utils.forward(
        shepp, GRID_SIZE, SPACE_STEP_SIZE, SOUND_SPEED, DETECTOR_DISTANCE, PML_WIDTH
    )
    shepp_measurements = shepp_measurements[:final_time_step, :]
    tikhonov_shepp = utils.tikhonov_reconstruction_reduced_time(
        shepp_measurements,
        1.0e-6,
        GRID_SIZE,
        PIC_SIZE,
        MAX_FREQUENCY,
        SPACE_STEP_SIZE,
        TIME_STEP_SIZE,
        SOUND_SPEED,
        DETECTOR_DISTANCE,
        PML_WIDTH,
        MAX_TIME_STEP,
    )
    reversal_shepp = utils.time_reversal(
        shepp_measurements,
        GRID_SIZE,
        PIC_SIZE,
        SOUND_SPEED,
        DETECTOR_DISTANCE,
        PML_WIDTH,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    images = [shepp, shepp_DST, tikhonov_shepp, reversal_shepp]

    for ax, img in zip(axes.flat, images):
        im = ax.imshow(img, cmap="Greys")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def method_comparison_reduced_time_disks(final_time_step=300):
    dst_functions = utils.generate_dst_basis(GRID_SIZE, MAX_FREQUENCY)
    disks = utils.disk_phantom(PIC_SIZE)
    disks_DST = utils.dst_representation(disks, dst_functions, SPACE_STEP_SIZE)
    disks = np.pad(
        disks,
        (
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
        ),
    )
    disks_DST = np.pad(
        disks_DST,
        (
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
            ((GRID_SIZE - PIC_SIZE) // 2, (GRID_SIZE - PIC_SIZE) // 2),
        ),
    )

    disks_measurements = utils.forward(
        disks, GRID_SIZE, SPACE_STEP_SIZE, SOUND_SPEED, DETECTOR_DISTANCE, PML_WIDTH
    )
    disks_measurements = disks_measurements[:final_time_step, :]
    tikhonov_disks = utils.tikhonov_reconstruction_reduced_time(
        disks_measurements,
        1.0e-6,
        GRID_SIZE,
        PIC_SIZE,
        MAX_FREQUENCY,
        SPACE_STEP_SIZE,
        TIME_STEP_SIZE,
        SOUND_SPEED,
        DETECTOR_DISTANCE,
        PML_WIDTH,
        MAX_TIME_STEP,
    )
    reversal_disks = utils.time_reversal(
        disks_measurements,
        GRID_SIZE,
        PIC_SIZE,
        SOUND_SPEED,
        DETECTOR_DISTANCE,
        PML_WIDTH,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    images = [disks, disks_DST, tikhonov_disks, reversal_disks]

    for ax, img in zip(axes.flat, images):
        im = ax.imshow(img, cmap="Greys")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # FOR EXAMPLE
    method_comparison_shepp()
    method_comparison_disks()
    method_comparison_reduced_time_shepp()
    method_comparison_reduced_time_disks()
