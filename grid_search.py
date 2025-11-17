from curlingsim import Stone, simulate
import numpy as np
import plotly.graph_objects as go


def create_grid(bd_low: tuple, bd_high: tuple, bd_steps: tuple):
    """Perform a grid search over the specified parameter bounds.
    Args:
        bd_low (tuple): Lower bounds for each parameter.
        bd_high (tuple): Upper bounds for each parameter.
        bd_steps (tuple): number of steps for each parameter.
    Returns:
        list of tuples: All combinations of parameters in the grid.
    """
    grids = []
    for i in range(len(bd_low)):
        min_val = bd_low[i]
        max_val = bd_high[i]
        step = bd_steps[i]

        values = np.linspace(min_val, max_val, step)
        grids.append(values)
    return np.array(np.meshgrid(*grids)).T.reshape(-1, len(bd_low))


def evaluate_parameters(params: tuple):
    """Evaluate a single set of parameters by simulating a curling throw.
    Args:
        params (tuple): A tuple containing (speed, angle, spin).
    Returns:
        float: The final distance from the button after simulation.
    """
    speed, angle, spin = params
    # Initialize a stone with the given parameters
    stone = Stone(x=0.0, y=1.37, v=speed, psi=np.radians(angle), omega=spin)

    # Simulate the stone's motion
    stones = [stone]
    _, out_detected = simulate(stones, dt=0.001, t_max=10.0)

    if out_detected[0]:
        return False

    button = (0.0, 34.747)
    sheet_width = 4.75
    xlim = (button[0] - sheet_width / 2.0, button[0] + sheet_width / 2.0)
    ylim = (button[1] - 6.401, button[1] + 1.829)

    if not (xlim[0] <= stone.x <= xlim[1]) or not (ylim[0] <= stone.y <= ylim[1]):
        # Stone is out of bounds, return False
        return False
    return True


if __name__ == "__main__":
    bd_low = (3.0, 60.0, -20.0)   # speed, angle, spin
    bd_high = (4.5, 120.0, 20.0)
    bd_steps = (20, 50, 50)

    param_grid = create_grid(bd_low, bd_high, bd_steps)
    print("Parameter combinations:")
    print(param_grid.shape)

    # Evaluate each parameter combination
    results = []
    for i, params in enumerate(param_grid):
        if i % 5000 == 0:
            print(f"Evaluating parameter set {i} / {len(param_grid)}")
        is_valid = evaluate_parameters(params)
        results.append((params, is_valid))

    # Plot valid and invalid results using Plotly
    valid_params = [r[0] for r in results if r[1]]
    invalid_params = [r[0] for r in results if not r[1]]
    valid_params = np.array(valid_params)
    invalid_params = np.array(invalid_params)

    fig = go.Figure()
    if len(valid_params) > 0:
        fig.add_trace(go.Scatter3d(
            x=valid_params[:, 0],
            y=valid_params[:, 1],
            z=valid_params[:, 2],
            mode='markers',
            marker=dict(size=4, color='green', opacity=0.3),
            name='Valid'
        ))
    # if len(invalid_params) > 0:
    #     fig.add_trace(go.Scatter3d(
    #         x=invalid_params[:, 0],
    #         y=invalid_params[:, 1],
    #         z=invalid_params[:, 2],
    #         mode='markers',
    #         marker=dict(size=4, color='red', opacity=0.3),
    #         name='Invalid'
    #     ))
    fig.update_layout(
        scene=dict(
            xaxis_title='Speed (m/s)',
            yaxis_title='Angle (degrees)',
            zaxis_title='Spin (rad/s)'
        ),
        title='Grid Search Results for Curling Parameters',
        legend=dict(x=0.8, y=0.9)
    )
    fig.show()