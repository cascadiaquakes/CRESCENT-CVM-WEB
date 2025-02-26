// Add grid lines to the study area.
let gridTransparency = 0.1; // Initial grid transparency value

function addGridLines(west, east, south, north) {
    const gridLineWidth = 0.5; // Line width for grid lines

    // Remove any existing grid lines before adding new ones
    viewer.entities.values.forEach(entity => {
        if (entity.name === "gridLine") {
            viewer.entities.remove(entity);
        }
    });

    // Draw vertical lines (longitude lines)
    for (let lon = Math.ceil(west); lon <= Math.floor(east); lon++) {
        const lonPositions = Cesium.Cartesian3.fromDegreesArray([
            lon, south,
            lon, north
        ]);
        viewer.entities.add({
            name: "gridLine",
            polyline: {
                positions: lonPositions,
                width: gridLineWidth,
                material: Cesium.Color.LIGHTGRAY.withAlpha(gridTransparency)
            }
        });
    }

    // Draw horizontal lines (latitude lines)
    for (let lat = Math.ceil(south); lat <= Math.floor(north); lat++) {
        const latPositions = Cesium.Cartesian3.fromDegreesArray([
            west, lat,
            east, lat
        ]);
        viewer.entities.add({
            name: "gridLine",
            polyline: {
                positions: latPositions,
                width: gridLineWidth,
                material: Cesium.Color.LIGHTGRAY.withAlpha(gridTransparency)
            }
        });
    }
}


