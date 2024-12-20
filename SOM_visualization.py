# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.path as mpath

# Define path to SOM dataset
nc_file = "/home/flo/LSP_analysis/Data/SOM_8_ssim_hgt_GRl_1900_2015/SOM_8_ssim_hgt_GRl_1900_2015"
# Open the NetCDF file using xarray
data = xr.open_dataset(nc_file)

# Extract the 'nodes' variable
nodes_data = data["nodes"]

n = len(nodes_data["n"])

# %% define map section

myProj = myProj = ccrs.NorthPolarStereo(central_longitude=-50)
myProj._threshold = myProj._threshold / 40.0  # for higher precision plot
noProj = ccrs.PlateCarree(central_longitude=0)


fig = plt.figure(figsize=(8, 12))
ax = fig.add_subplot(1, 1, 1, projection=myProj)

[ax_hdl] = ax.plot(
    [-100, -75, -50, -25, -1, -1, -1, -1, -25, -50, -75, -100, -100, -100],
    [55, 55, 55, 55, 55, 55, 90, 90, 90, 90, 90, 90, 90, 55],
    color="black",
    linewidth=0.5,
    transform=noProj,
)
tx_path = ax_hdl._get_transformed_path()
path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
polygon1s = mpath.Path(path_in_data_coords.vertices)

# %% plot FIG 3

params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (18, 9),
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "lines.linewidth": 4,
    "lines.markersize": 10,
    "hatch.linewidth": 0.1,
}
plt.rcParams.update(params)

fig, axs = plt.subplots(
    3, 3, figsize=(15, 10), subplot_kw={"projection": myProj}
)  

# Plot each node data on separate subplots using contourf
for i in range(n):
    if i < 3:
        ax = axs[0, i]

    elif i < 6:
        ax = axs[1, i - 3]

    else:
        ax = axs[2, i - 6]

    ax.coastlines()

    im = ax.contourf(
        nodes_data["lon"],
        nodes_data["lat"],
        nodes_data[i, :, :],
        transform=ccrs.PlateCarree(),
        cmap=plt.cm.PuOr_r,
        norm=mpl.colors.Normalize(vmin=5000, vmax=5600),
    )
    ax.set_boundary(polygon1s)
    ax.scatter(
        -51.128333,
        71.140556,
        edgecolor="k",
        s=100,
        c="green",
        zorder=14,
        transform=ccrs.PlateCarree(),
        label="study site",
    )
    norm = mpl.colors.Normalize(vmin=5000, vmax=5600)
    cb1 = plt.cm.ScalarMappable(
        cmap=plt.cm.PuOr_r, norm=mpl.colors.Normalize(vmin=5000, vmax=5600)
    )

    gl = ax.gridlines(draw_labels=False)

    gl.ylocator = plt.MaxNLocator(4)

    ax.text(
        -104,
        60,
        f"LSP {i+1}",
        fontweight="bold",
        transform=ccrs.PlateCarree(),
        fontsize=20,
        rotation=40,
    )

    ax.text(-84, 48.6, "80°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=-30)
    ax.text(-64, 51, "60°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=-8)
    ax.text(-44, 52, "40°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=15)
    ax.text(-24, 53, "20°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=30)

    ax.text(-20, 80, "80°N", transform=ccrs.PlateCarree(), fontsize=18, rotation=-40)
    ax.text(-10, 70, "70°N", transform=ccrs.PlateCarree(), fontsize=18, rotation=-40)
    ax.text(-7, 60, "60°N", transform=ccrs.PlateCarree(), fontsize=18, rotation=-40)

h, l = ax.get_legend_handles_labels()

cax = fig.add_axes([0.62, 0.25, 0.25, 0.02])
cax.legend(h, l, loc=[0.15, -8.0], fontsize=18, frameon=False)

cbar = fig.colorbar(cb1, cax=cax, orientation="horizontal", extend="both")
cbar.set_label("geopotential height\n500hPa [gpm]", labelpad=15)

axs[2, 2].remove()
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()


# %%
