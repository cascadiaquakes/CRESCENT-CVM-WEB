import os
import xarray as xr


def extract_plot_data(items_dict):
    try:
        plot_data = xr.load_dataset(
            os.path.join("static", "netcdf", items_dict["data_file"])
        )
        start_lat = float(items_dict["start_lat"])
        start_lon = float(items_dict["start_lng"])
        end_lat = float(items_dict["end_lat"])
        end_lon = float(items_dict["end_lng"])
        depth = float(items_dict["start_depth"])
        lon_points = int(items_dict["lng_points"])
        lat_points = int(items_dict["lat_points"])
        plot_grid_mapping = items_dict["plot_grid_mapping"]
        units = items_dict["units"]

        unit_standard, depth_factor = unit_conversion_factor(
            plot_data["depth"].attrs["units"], units
        )
        if depth_factor != 1:
            new_depth_values = plot_data["depth"] * depth_factor
            new_depth = xr.DataArray(
                new_depth_values, dims=["depth"], attrs=plot_data["depth"].attrs
            )
            new_depth.attrs["units"] = unit_standard
            new_depth.attrs["long_name"] = (
                f"{new_depth.attrs['standard_name']} [{unit_standard}]"
            )
            plot_data = plot_data.assign_coords(depth=new_depth)
        else:
            plot_data.depth.attrs["units"] = unit_standard
            plot_data.depth.attrs["long_name"] = (
                f"{plot_data.depth.attrs['standard_name']} ({unit_standard})"
            )

        vmin = items_dict["start_value"].strip()
        vmax = items_dict["end_value"].strip()

        plot_var = items_dict["plot_variable"]
        plot_data = plot_data[plot_var]
        unit_standard, var_factor = unit_conversion_factor(
            plot_data.attrs["units"], units
        )
        plot_data.data *= var_factor
        plot_data.attrs["units"] = unit_standard
        plot_data.attrs["display_name"] = (
            f"{plot_data.attrs['long_name']} [{unit_standard}]"
        )

        return (
            plot_data,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            depth,
            lon_points,
            lat_points,
            plot_grid_mapping,
            vmin,
            vmax,
        )

    except Exception as ex:
        logging.error(f"[ERR] {ex}")
        raise


def plot_depth_slice(
    plot_data,
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    depth,
    lon_points,
    lat_points,
    plot_grid_mapping,
    vmin,
    vmax,
    items_dict,
):
    try:
        image_width = float(items_dict["image_width"])
        image_height = float(items_dict["image_height"])
        figsize = (image_width, image_height)
        dpi = 100
        title = items_dict["title"]
        cmap = items_dict["colormap"]
        interp_type = items_dict["interpolation_method"]
        utm_zone = None

        meta = plot_data.attrs
        if "grid_mapping_name" not in meta:
            grid_mapping_name = "latitude_longitude"
        else:
            grid_mapping_name = meta["grid_mapping_name"]

        if "utm_zone" in meta:
            utm_zone = meta["utm_zone"]
            ellipsoid = meta["ellipsoid"]

        if grid_mapping_name == "latitude_longitude":
            lon_list = np.linspace(start_lon, end_lon, lon_points).tolist()
            lat_list = np.linspace(start_lat, end_lat, lat_points).tolist()
            x_list, y_list = project_lonlat_utm(lon_list, lat_list, utm_zone, ellipsoid)
            x_grid, y_grid = np.meshgrid(lon_list, lat_list)
        else:
            start_x, start_y = project_lonlat_utm(
                start_lon, start_lat, utm_zone, ellipsoid
            )
            end_x, end_y = project_lonlat_utm(end_lon, end_lat, utm_zone, ellipsoid)
            x_list = np.linspace(start_x, end_x, lon_points).tolist()
            y_list = np.linspace(start_y, end_y, lat_points).tolist()
            lon_list, lat_list = project_lonlat_utm(
                x_list, y_list, utm_zone, ellipsoid, xy_to_latlon=True
            )
            x_grid, y_grid = np.meshgrid(x_list, y_list)

        interpolated_values_2d = np.zeros_like(x_grid)
        if interp_type == "none":
            start_lat = closest(plot_data["latitude"].data, start_lat)
            end_lat = closest(plot_data["latitude"].data, end_lat)
            start_lon = closest(plot_data["longitude"].data, start_lon)
            end_lon = closest(plot_data["longitude"].data, end_lon)
            depth_closest = closest(list(plot_data["depth"].data), depth)
            plot_data = plot_data.where(plot_data.depth == depth_closest, drop=True)
            plot_data = plot_data.where(
                (plot_data.latitude >= start_lat)
                & (plot_data.latitude <= end_lat)
                & (plot_data.longitude >= start_lon)
                & (plot_data.longitude <= end_lon),
                drop=True,
            )

            if plot_grid_mapping == "latitude_longitude":
                aspect_ratio = 1 / math.cos(math.radians((start_lat + end_lat) / 2))
            else:
                aspect_ratio = 1

            x2 = grid_mapping_dict[plot_grid_mapping]["x"]
            y2 = grid_mapping_dict[plot_grid_mapping]["y"]
            fig_width, fig_height = figsize
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

            if plot_grid_mapping == "latitude_longitude":
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                if vmin and vmax:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                    )
                elif vmin:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        vmin=float(vmin),
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                    )
                elif vmax:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        vmax=float(vmax),
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                    )
                else:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                    )

                plt.xticks(rotation=45)
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle=":")
                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False
            else:
                if vmin and vmax:
                    plot_data.plot(
                        figsize=figsize, cmap=cmap, vmin=vmin, vmax=vmax, x=x2, y=y2
                    )
                elif vmin:
                    plot_data.plot(
                        figsize=figsize, cmap=cmap, vmin=float(vmin), x=x2, y=y2
                    )
                elif vmax:
                    plot_data.plot(
                        figsize=figsize, cmap=cmap, vmax=float(vmax), x=x2, y=y2
                    )
                else:
                    plot_data.plot(figsize=figsize, cmap=cmap, x=x2, y=y2)

                plt.xticks(rotation=45)
        else:
            if grid_mapping_name == "latitude_longitude":
                interpolated_values_2d = plot_data.interp(
                    latitude=lat_list,
                    longitude=lon_list,
                    depth=depth,
                    method=interp_type,
                )
            else:
                interpolated_values_2d = plot_data.interp(
                    y=y_list, x=x_list, depth=depth, method=interp_type
                )

            if np.all(np.isnan(interpolated_values_2d)) and interp_type == "cubic":
                message = f"[ERR] Data with NaN values. Can't use the cubic spline interpolation"
                logging.error(message)
                return Response(
                    content=create_error_image(message), media_type="image/png"
                )

            if grid_mapping_name == plot_grid_mapping:
                x_label = plot_data[grid_mapping_dict[plot_grid_mapping]["x"]].attrs[
                    "long_name"
                ]
                y_label = plot_data[grid_mapping_dict[plot_grid_mapping]["y"]].attrs[
                    "long_name"
                ]
            else:
                x_label = plot_data[grid_mapping_dict[plot_grid_mapping]["x2"]].attrs[
                    "long_name"
                ]
                y_label = plot_data[grid_mapping_dict[plot_grid_mapping]["y2"]].attrs[
                    "long_name"
                ]

            if plot_grid_mapping == "latitude_longitude":
                aspect_ratio = 1 / math.cos(math.radians((start_lat + end_lat) / 2))
            else:
                aspect_ratio = 1

            fig_width, fig_height = figsize
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            if plot_grid_mapping == "latitude_longitude":
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            levels = np.linspace(vmin, vmax, 15)
            if plot_grid_mapping == "latitude_longitude":
                contourf = ax.contourf(
                    lon_list,
                    lat_list,
                    interpolated_values_2d,
                    levels=levels,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                )
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS)
                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False
            else:
                contourf = plt.contourf(
                    x_list, y_list, interpolated_values_2d, levels=levels, cmap=cmap
                )

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            cbar = plt.colorbar(contourf, ax=plt.gca())
            cbar.set_label(plot_data.attrs["display_name"])

        plt.xticks(rotation=45)
        plt.title(title)

        ax.set_aspect(aspect_ratio)
        if os.path.isfile(LOGO_FILE):
            plot_ax = fig.axes[0]
            logo = plt.imread(LOGO_FILE)
            aspect_ratio = logo.shape[1] / logo.shape[0]
            logo_height_inches = 0.3
            logo_width_inches = logo_height_inches * aspect_ratio
            norm_logo_width = logo_width_inches / fig_width
            norm_logo_height = logo_height_inches / fig_height
            norm_offset_below = 2 * LOGO_INCH_OFFSET_BELOW / fig_height
            logo_bottom = (
                plot_ax.get_position().ymin - norm_logo_height - norm_offset_below
            )
            logo_left = plot_ax.get_position().xmin
            logo_ax = fig.add_axes(
                [logo_left, logo_bottom, norm_logo_width, norm_logo_height]
            )
            logo_ax.imshow(logo)
            logo_ax.axis("off")
            text_y_position = -logo_bottom - norm_logo_height
            logo_ax.text(
                logo_left,
                text_y_position,
                f"{items_dict['version_label']}\n",
                ha="left",
                fontsize=6,
                color="#004F59",
            )
        else:
            logging.warning(f"[WARN] Logo file not found: {LOGO_FILE}")

        plot_buf = BytesIO()
        plt.savefig(plot_buf, format="png")
        plot_buf.seek(0)
        base64_plot = base64.b64encode(plot_buf.getvalue()).decode("utf-8")
        plt.clf()
        try:
            if plot_data.size > 1000000:
                raise ValueError("Data too large")
            csv_buf = BytesIO()
            plot_data.to_dataframe().to_csv(csv_buf)
            csv_buf.seek(0)
            base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")
        except ValueError as e:
            csv_buf = BytesIO()
            csv_buf.write(b"Data too large")
            csv_buf.seek(0)
            base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")
        except Exception as e:
            csv_buf = BytesIO()
            error_message = f"An error occurred: {str(e)}"
            csv_buf.write(error_message.encode())
            csv_buf.seek(0)
            base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

        content = {"image": base64_plot, "csv_data": base64_csv}

        plot_data.close()
        del plot_data
        gc.collect()

        return JSONResponse(content=content)
    except Exception as ex:
        logging.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            ),
            media_type="image/png",
        )


@router.post("/depth-slice")
async def depthslice(request: Request):
    form_data = await request.form()
    items_dict = dict(form_data.items())
    logging.info(f"[INFO] POST: {items_dict}")

    try:
        (
            plot_data,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            depth,
            lon_points,
            lat_points,
            plot_grid_mapping,
            vmin,
            vmax,
        ) = extract_plot_data(items_dict)
        return plot_depth_slice(
            plot_data,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            depth,
            lon_points,
            lat_points,
            plot_grid_mapping,
            vmin,
            vmax,
            items_dict,
        )
    except Exception as ex:
        logging.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            ),
            media_type="image/png",
        )
