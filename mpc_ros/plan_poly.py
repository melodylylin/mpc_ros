import numpy as np
import minsnap_trajectories as ms
import matplotlib.pyplot as plt

def plan_poly_traj(boundary_cdn, tf, sample_time):
    polys = ms.generate_trajectory(
        boundary_cdn,
        degree=8,  # Polynomial degree
        idx_minimized_orders=(3, 4),  
        num_continuous_orders=3,  
        algorithm="closed-form",  # Or "constrained"
    )


    # Inspect the output
    t = polys.time_reference
    dt = polys.durations
    cfs = polys.coefficients
    t = np.linspace(0, tf, int(tf/sample_time))
    #  Sample up to the 3rd order (acceleration) -----v
    pva = ms.compute_trajectory_derivatives(polys, t, 3)
    position, velocity, *_ = pva

    return position, velocity

def main():
    boundary_cdn = [
        ms.Waypoint(
            time=0.0,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.5, 0.0]),
        ),
        ms.Waypoint(  # Any higher-order derivatives
            time=8.0,
            position=np.array([5.0, -5.0]),
            velocity=np.array([0.5, -0.5]),
            acceleration=np.array([0.0, 0.0]),
        ),
        ms.Waypoint(  # Potentially leave intermediate-order derivatives unspecified
            time=16.0,
            position=np.array([10.0, -10.0]),
            velocity=np.array([0.5, 0.0]),
        ),
    ]
    position, velocity = plan_poly_traj(boundary_cdn, 16, 0.1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(position[:, 0], position[:, 1], label="Position Trajectory")

    position_waypoints = np.array([it.position for it in boundary_cdn])
    ax.plot(
        position_waypoints[:, 0],
        position_waypoints[:, 1],
        "ro",
        label="Position Waypoints",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    main()