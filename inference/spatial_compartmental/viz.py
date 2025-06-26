import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import arviz as az
import numpyro
import matplotlib.ticker as ticker
from abm.spatial_compartmental.utils import calc_start_n_initial


def SI_stoch_viz(
    ax,
    abm,
    p_infect,
    prop_initial_infected,
    num_steps,
    grid_sizes,
    colours,
    abm_keys,
    mean_or_median="median",
    ci_alpha=0.1,
    beautify_spines=False,
    xlim=None,
):
    assert len(grid_sizes) == len(colours), "Grid sizes must match number of colours"
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticks(range(0, int(num_steps) + 1))
    # ax.set_xlim(0, num_steps) if xlim is None else ax.set_xlim(*xlim)
    for grid_size, colour in zip(grid_sizes, colours):
        _, _, I_list = abm(
            abm_keys,
            grid_size,
            num_steps,
            p_infect,
            calc_start_n_initial(prop_initial_infected, grid_size),
        )
        I_list /= grid_size * grid_size
        line_data = (
            jnp.quantile(I_list, 0.50, axis=0)
            if mean_or_median.lower()
            == "median"  # NOTE: Flaky, should explicitly check if mean
            else jnp.mean(I_list, axis=0)
        )
        print(grid_size, line_data[0])
        quantiles_I_2pt5 = jnp.quantile(I_list, 0.025, axis=0)
        quantiles_I_97pt5 = jnp.quantile(I_list, 0.975, axis=0)
        ax.plot(range(num_steps + 1), line_data, color=colour)
        ax.fill_between(
            range(num_steps + 1),
            quantiles_I_2pt5,
            quantiles_I_97pt5,
            color=colour,
            alpha=ci_alpha,
        )
    if beautify_spines:
        ax.spines[["right", "top"]].set_visible(False)
    return ax


def posterior_comparison(
    ax,
    abm_config,
    abm_posterior,
    surrogates_posterior,
    target_param_and_label,
    rhats,
    prior=None,
    title="ABM vs Surrogate Posterior",
    xlim=(0, 1),
    ylim=None,
    logdensities=False,
    show_legend=True,
    set_x_and_ylabels=True,
):
    # TODO: Fix for when data is truncated
    target_param, target_label = target_param_and_label

    if prior is not None:
        sns.kdeplot(
            prior[target_label],
            fill=True,
            color="lightgreen",
            ax=ax,
            alpha=0.2,
            label="Prior",
            clip=(0, None),
            # clip=(0, None),
        )
    sns.kdeplot(
        abm_posterior[abm_config]["posterior_samples"][target_label],
        fill=True,
        color="xkcd:lavender",
        ax=ax,
        label="ABM Posterior",
        hatch="\\" if rhats["abm"] >= 1.1 else "",
    )
    for surrogate_name, surrogate_samples in surrogates_posterior.items():
        sns.kdeplot(
            surrogate_samples[abm_config]["posterior_samples"][target_label],
            fill=True,
            ax=ax,
            label=f"{surrogate_name} Posterior",
            hatch="\\" if rhats[surrogate_name] >= 1.1 else "",
        )
    ax.axvline(
        x=target_param, color="red", linestyle="--", label=f"True {target_label}"
    )

    if show_legend:
        ax.legend()
    ax.set_xlim(xlim[0], xlim[1])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if set_x_and_ylabels:
        ax.set_xlabel("Parameter Value")
    else:
        ax.set_ylabel("")
    if logdensities:
        ax.set_yscale("log")
    return ax


# TODO: Extend for SIR
def post_prediction_comparison(
    ax,
    abm_config,
    abm_posterior,
    surrogates_posterior,
    data,
    y_label,
    hpdi_alpha=0.3,
    show_legend=True,
    set_x_and_ylabels=True,
):
    x = jnp.arange(len(data[abm_config]))
    abm_pred_mean = jnp.mean(abm_posterior[abm_config]["obs"], axis=0)
    abm_pred_hpdi = numpyro.diagnostics.hpdi(abm_posterior[abm_config]["obs"], 0.95)

    ax.plot(
        x, abm_pred_mean, color="xkcd:purple", label="ABM Posterior Predictive Mean"
    )
    ax.fill_between(
        x,
        abm_pred_hpdi[0],
        abm_pred_hpdi[1],
        alpha=hpdi_alpha,
        interpolate=True,
        color="xkcd:lavender",
        label="ABM 95% HPDI",
    )

    for surrogate_name, surrogate_samples in surrogates_posterior.items():
        # print(surrogate_samples[abm_config]["obs"].shape)
        surr_pred_mean = jnp.mean(
            surrogate_samples[abm_config]["obs"].squeeze(), axis=0
        )
        surr_pred_hpdi = numpyro.diagnostics.hpdi(
            surrogate_samples[abm_config]["obs"].squeeze(), 0.95
        )
        ax.plot(
            x,
            surr_pred_mean,
            label=f"{surrogate_name} Posterior Predictive Mean",
        )
        ax.fill_between(
            x,
            surr_pred_hpdi[0],
            surr_pred_hpdi[1],
            alpha=hpdi_alpha,
            interpolate=True,
            label=f"{surrogate_name} 95% HPDI",
        )
    ax.plot(x, data[abm_config], "o", color="red", label="Data")
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.set_xlim(0, None)
    if set_x_and_ylabels:
        ax.set_xlabel("Timestep")
        ax.set_ylabel(y_label)
    if show_legend:
        ax.legend()
    return ax
