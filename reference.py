import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

N_ITERATIONS = 15_000
REYNOLDS_NUMBER = 80

N_POINTS_X = 300
N_POINTS_Y = 50

CYLINDER_CENTER_INDEX_X = N_POINTS_X // 5
CYLINDER_CENTER_INDEX_Y = N_POINTS_Y // 2
CYLINDER_RADIUS_INDICES = N_POINTS_Y // 9

MAX_HORIZONTAL_INFLOW_VELOCITY = 0.04

VISUALIZE = True
PLOT_EVERY_N_STEPS = 100
SKIP_FIRST_N_ITERATIONS = 5000


r"""
LBM Grid: D2Q9
    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8
"""

N_DISCRETE_VELOCITIES = 9

LATTICE_VELOCITIES = jnp.array([
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1,],
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1,]
])

LATTICE_INDICES = jnp.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8,
])

OPPOSITE_LATTICE_INDICES = jnp.array([
    0, 3, 4, 1, 2, 7, 8, 5, 6,
])

LATTICE_WEIGHTS = jnp.array([
    4/9,                        # Center Velocity [0,]
    1/9,  1/9,  1/9,  1/9,      # Axis-Aligned Velocities [1, 2, 3, 4]
    1/36, 1/36, 1/36, 1/36,     # 45 ° Velocities [5, 6, 7, 8]
])

RIGHT_VELOCITIES = jnp.array([1, 5, 8])
UP_VELOCITIES = jnp.array([2, 5, 6])
LEFT_VELOCITIES = jnp.array([3, 6, 7])
DOWN_VELOCITIES = jnp.array([4, 7, 8])
PURE_VERTICAL_VELOCITIES = jnp.array([0, 2, 4])
PURE_HORIZONTAL_VELOCITIES = jnp.array([0, 1, 3])


def get_density(discrete_velocities):
    density = jnp.sum(discrete_velocities, axis=-1)

    return density

def get_macroscopic_velocities(discrete_velocities, density):
    macroscopic_velocities = jnp.einsum(
        "NMQ,dQ->NMd",
        discrete_velocities,
        LATTICE_VELOCITIES,
    ) / density[..., jnp.newaxis]

    return macroscopic_velocities

def get_equilibrium_discrete_velocities(macroscopic_velocities, density):
    projected_discrete_velocities = jnp.einsum(
        "dQ,NMd->NMQ",
        LATTICE_VELOCITIES,
        macroscopic_velocities,
    )
    macroscopic_velocity_magnitude = jnp.linalg.norm(
        macroscopic_velocities,
        axis=-1,
        ord=2,
    )
    equilibrium_discrete_velocities = (
        density[..., jnp.newaxis]
        *
        LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
        *
        (
            1
            +
            3 * projected_discrete_velocities
            +
            9/2 * projected_discrete_velocities**2
            -
            3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]**2
        )
    )

    return equilibrium_discrete_velocities

def main():
    jax.config.update("jax_enable_x64", True)

    kinematic_viscosity = (
        (
            MAX_HORIZONTAL_INFLOW_VELOCITY
            *
            CYLINDER_RADIUS_INDICES
        ) / (
            REYNOLDS_NUMBER
        )
    )
    relaxation_omega = (
        (
            1.0
        ) / (
            3.0
            *
            kinematic_viscosity
            +
            0.5
        )
    )

    # Define a mesh
    x = jnp.arange(N_POINTS_X)
    y = jnp.arange(N_POINTS_Y)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Obstacle Mask: An array of the shape like X or Y, but contains True if the
    # point belongs to the obstacle and False if not
    obstacle_mask = (
        jnp.sqrt(
            (
                X
                -
                CYLINDER_CENTER_INDEX_X
            )**2
            +
            (
                Y
                -
                CYLINDER_CENTER_INDEX_Y
            )**2
        )
        <
            CYLINDER_RADIUS_INDICES
    )

    velocity_profile = jnp.zeros((N_POINTS_X, N_POINTS_Y, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(MAX_HORIZONTAL_INFLOW_VELOCITY)

    @jax.jit
    def update(discrete_velocities_prev):
        # (1) Prescribe the outflow BC on the right boundary
        discrete_velocities_prev = discrete_velocities_prev.at[-1, :, LEFT_VELOCITIES].set(
            discrete_velocities_prev[-2, :, LEFT_VELOCITIES]
        )

        # (2) Macroscopic Velocities
        density_prev = get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = get_macroscopic_velocities(
            discrete_velocities_prev,
            density_prev,
        )

        # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme
        macroscopic_velocities_prev =\
            macroscopic_velocities_prev.at[0, 1:-1, :].set(
                velocity_profile[0, 1:-1, :]
            )
        density_prev = density_prev.at[0, :].set(
            (
                get_density(discrete_velocities_prev[0, :, PURE_VERTICAL_VELOCITIES].T)
                +
                2 *
                get_density(discrete_velocities_prev[0, :, LEFT_VELOCITIES].T)
            ) / (
                1 - macroscopic_velocities_prev[0, :, 0]
            )
        )

        # (4) Compute discrete Equilibria velocities
        equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
            macroscopic_velocities_prev,
            density_prev,
        )

        # (3) Belongs to the Zou/He scheme
        discrete_velocities_prev = \
            discrete_velocities_prev.at[0, :, RIGHT_VELOCITIES].set(
                equilibrium_discrete_velocities[0, :, RIGHT_VELOCITIES]
            )

        # (5) Collide according to BGK
        discrete_velocities_post_collision = (
            discrete_velocities_prev
            -
            relaxation_omega
            *
            (
                discrete_velocities_prev
                -
                equilibrium_discrete_velocities
            )
        )

        # (6) Bounce-Back Boundary Conditions to enfore the no-slip
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_post_collision =\
                discrete_velocities_post_collision.at[obstacle_mask, LATTICE_INDICES[i]].set(
                    discrete_velocities_prev[obstacle_mask, OPPOSITE_LATTICE_INDICES[i]]
                )

        # (7) Stream alongside lattice velocities
        discrete_velocities_streamed = discrete_velocities_post_collision
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_streamed = discrete_velocities_streamed.at[:, :, i].set(
                jnp.roll(
                    jnp.roll(
                        discrete_velocities_post_collision[:, :, i],
                        LATTICE_VELOCITIES[0, i],
                        axis=0,
                    ),
                    LATTICE_VELOCITIES[1, i],
                    axis=1,
                )
            )

        return discrete_velocities_streamed


    discrete_velocities_prev = get_equilibrium_discrete_velocities(
        velocity_profile,
        jnp.ones((N_POINTS_X, N_POINTS_Y)),
    )

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 6), dpi=100)

    for iteration_index in tqdm(range(N_ITERATIONS)):
        discrete_velocities_next = update(discrete_velocities_prev)

        discrete_velocities_prev = discrete_velocities_next

        if iteration_index % PLOT_EVERY_N_STEPS == 0 and VISUALIZE and iteration_index > SKIP_FIRST_N_ITERATIONS:
            density = get_density(discrete_velocities_next)
            macroscopic_velocities = get_macroscopic_velocities(
                discrete_velocities_next,
                density,
            )
            velocity_magnitude = jnp.linalg.norm(
                macroscopic_velocities,
                axis=-1,
                ord=2,
            )
            d_u__d_x, d_u__d_y = jnp.gradient(macroscopic_velocities[..., 0])
            d_v__d_x, d_v__d_y = jnp.gradient(macroscopic_velocities[..., 1])
            curl = (d_u__d_y - d_v__d_x)

            # Velocity Magnitude Contour Plot in the top
            plt.subplot(211)
            plt.contourf(
                X,
                Y,
                velocity_magnitude,
                levels=50,
                cmap=cmr.amber,
            )
            plt.colorbar().set_label("Velocity Magnitude")
            plt.gca().add_patch(plt.Circle(
                (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDICES,
                color="darkgreen",
            ))

            # Vorticity Magnitude Contour PLot in the bottom
            plt.subplot(212)
            plt.contourf(
                X,
                Y,
                curl,
                levels=50,
                cmap=cmr.redshift,
                vmin=-0.02,
                vmax= 0.02,
            )
            plt.colorbar().set_label("Vorticity Magnitude")
            plt.gca().add_patch(plt.Circle(
                (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDICES,
                color="darkgreen",
            ))

            plt.draw()
            plt.pause(0.0001)
            plt.clf()

    if VISUALIZE:
        plt.show()



if __name__ == "__main__":
    main()