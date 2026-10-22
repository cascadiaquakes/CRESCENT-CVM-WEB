// Add grid lines to the study area.
let gridTransparency = 0.1; // Initial grid transparency value
let gridVisible = true;

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
            show: gridVisible,
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
            show: gridVisible,
            polyline: {
                positions: latPositions,
                width: gridLineWidth,
                material: Cesium.Color.LIGHTGRAY.withAlpha(gridTransparency)
            }
        });
    }
}

// Show or hide the grid lines without redrawing them.
function setGridVisible(show) {
    gridVisible = show;
    viewer.entities.values.forEach(entity => {
        if (entity.name === "gridLine") {
            entity.show = show;
        }
    });
}

// GeoJSON data via local storage
function handleLocalGeoJsonData(checkboxId, dataArray, labelArray, colorArray, lineWidthArray, fillOpacityArray, dataSourcesArray, extraLogic) {
    var checkbox = document.getElementById(checkboxId);
    function handleData(checked) {
        if (checked) {
            for (var i = 0; i < dataArray.length; i++) {
                (function (index) {
                    Cesium.GeoJsonDataSource.load(dataArray[index], {
                        label: labelArray[index],
                        stroke: colorArray[index],
                        fill: colorArray[index].withAlpha(fillOpacityArray[index]),
                        strokeWidth: lineWidthArray[index],
                        markerSymbol: '?'
                    }).then(function (dataSource) {
                        if (extraLogic) {
                            extraLogic(dataSource, index);
                        }

                        dataSourcesArray.push(dataSource);
                        viewer.dataSources.add(dataSource);
                    }).catch(function (error) {
                        console.error('Error loading GeoJSON:', error);
                    });
                })(i);
            }
        } else {
            for (var i = 0; i < dataSourcesArray.length; i++) {
                if (viewer.dataSources.contains(dataSourcesArray[i])) {
                    viewer.dataSources.remove(dataSourcesArray[i]);
                }
            }
            dataSourcesArray.length = 0; // Clear the array
        }
    }

    checkbox.addEventListener('change', function () {
        handleData(this.checked);
    });

    handleData(checkbox.checked);
}

// GeoJSON data via S3 bucket
function handleS3GeoJsonData(checkboxId, dataArray, labelArray, colorArray, lineWidthArray, fillOpacityArray, dataSourcesArray, extraLogic) {
    var checkbox = document.getElementById(checkboxId);

    function handleData(checked) {
        if (checked) {
            for (var i = 0; i < dataArray.length; i++) {
                (function (index) {
                    Cesium.GeoJsonDataSource.load(`/data/fetch_json_s3?file_name=${encodeURIComponent(dataArray[index])}&prefix_key=geojson`, {
                        label: labelArray[index],
                        stroke: colorArray[index],
                        fill: colorArray[index].withAlpha(fillOpacityArray[index]),
                        strokeWidth: lineWidthArray[index],
                        markerSymbol: '?'
                    }).then(function (dataSource) {
                        if (extraLogic) {
                            extraLogic(dataSource, index);
                        }

                        dataSourcesArray.push(dataSource);
                        viewer.dataSources.add(dataSource);
                    }).catch(function (error) {
                        console.error('Error loading GeoJSON:', error);
                    });
                })(i);
            }
        } else {
            for (var i = 0; i < dataSourcesArray.length; i++) {
                if (viewer.dataSources.contains(dataSourcesArray[i])) {
                    viewer.dataSources.remove(dataSourcesArray[i]);
                }
            }
            dataSourcesArray.length = 0; // Clear the array
        }
    }

    checkbox.addEventListener('change', function () {
        handleData(this.checked);
    });

    handleData(checkbox.checked);
}

// Function to enforce a maximum of 2 selections
function enforceSelectionLimit() {
    const dropdown = document.getElementById('select2dSurface');
    const selectedOptions = Array.from(dropdown.selectedOptions);

    // Enforce selection limit of 2
    if (selectedOptions.length > 2) {
        // Deselect the latest selected option
        selectedOptions[selectedOptions.length - 1].selected = false;
        alert('You can only select a maximum of 2 options.');
    }
}

// Draw a legend symbol that matches how the layer appears on the map:
// 'fill' (default square), 'outline' (faint box with a border), 'circle' or 'line'.
function setLegendSymbol(el, color, shape) {
    const css = color.toCssColorString();
    el.style.backgroundColor = css;
    if (shape === 'circle') {
        el.style.borderRadius = '50%';
    } else if (shape === 'line') {
        el.style.width = '14px';
        el.style.height = '3px';
        el.style.border = 'none';
        el.style.borderRadius = '1px';
    } else if (shape === 'outline') {
        el.style.backgroundColor = color.withAlpha(0.15).toCssColorString();
        el.style.border = '2px solid ' + css;
        el.style.boxSizing = 'border-box';
    }
}

// "Satellite imagery" and "Hide below ground" checkboxes. Occlusion needs the terrain
// surface, which only renders while the globe is shown, so it is off and disabled until
// imagery is on. The scene only redraws on request, so ask for a frame after each change.
function wireImageryToggles(viewer) {
    const globeBox = document.getElementById('toggleGlobeCheckbox');
    const occlusion = document.getElementById('toggleTerrainOcclusionCheckbox');
    function sync() {
        viewer.scene.globe.show = globeBox.checked;
        occlusion.disabled = !globeBox.checked;
        viewer.scene.globe.depthTestAgainstTerrain = globeBox.checked && occlusion.checked;
        occlusion.parentElement.title = globeBox.checked
            ? 'Hide anything below the terrain surface'
            : 'Turn on Satellite imagery to use this';
        viewer.scene.requestRender();
    }
    globeBox.addEventListener('change', sync);
    occlusion.addEventListener('change', sync);
    sync();
}

// requestRenderMode only repaints on camera moves, so layer toggles and layers that
// finish loading in the background don't show until the map is touched. Repaint on
// any control change, and keep repainting while entity geometry is still being built.
function redrawOnChanges(viewer) {
    const scene = viewer.scene;
    const redraw = () => scene.requestRender();
    document.addEventListener('change', redraw, true);
    document.addEventListener('input', redraw, true);
    viewer.entities.collectionChanged.addEventListener(redraw);
    viewer.dataSources.dataSourceAdded.addEventListener(redraw);
    viewer.dataSources.dataSourceRemoved.addEventListener(redraw);
    viewer.clock.onTick.addEventListener(function () {
        if (!viewer.dataSourceDisplay.ready) redraw();
    });
}

// Rotate the view so north points up, keeping the same spot centered and the same tilt.
function snapNorthUp() {
    const scene = viewer.scene;
    const camera = viewer.camera;
    const center = new Cesium.Cartesian2(scene.canvas.clientWidth / 2, scene.canvas.clientHeight / 2);
    const target = camera.pickEllipsoid(center, scene.globe.ellipsoid);
    if (!target) {
        camera.flyTo({
            destination: camera.positionWC,
            orientation: { heading: 0.0, pitch: camera.pitch, roll: 0.0 },
            duration: 0.5
        });
        return;
    }
    // Measure the tilt at the target, not the camera, so repeat clicks don't add tilt.
    const toCamera = Cesium.Cartesian3.subtract(camera.positionWC, target, new Cesium.Cartesian3());
    const range = Cesium.Cartesian3.magnitude(toCamera);
    const up = scene.globe.ellipsoid.geodeticSurfaceNormal(target, new Cesium.Cartesian3());
    const pitch = -Math.asin(Cesium.Cartesian3.dot(Cesium.Cartesian3.normalize(toCamera, toCamera), up));
    camera.flyToBoundingSphere(new Cesium.BoundingSphere(target, 0.0), {
        offset: new Cesium.HeadingPitchRange(0.0, pitch, range),
        duration: 0.5
    });
}

// Let users drag an overlay (the north arrow) off whatever it covers.
// A click without a drag calls onClick, if given.
function makeDraggable(el, onClick) {
    let offsetX = 0, offsetY = 0, startX = 0, startY = 0;
    el.title = onClick ? 'Click to face north, drag to move' : 'Drag to move';
    el.style.pointerEvents = 'auto';
    el.style.cursor = 'grab';
    el.style.touchAction = 'none';
    el.addEventListener('pointerdown', function (e) {
        // offsetLeft/Top ignore the heading rotation, so the arrow doesn't jump on grab.
        offsetX = e.clientX - el.offsetLeft;
        offsetY = e.clientY - el.offsetTop;
        startX = e.clientX;
        startY = e.clientY;
        el.setPointerCapture(e.pointerId);
        el.style.cursor = 'grabbing';
        e.preventDefault();
    });
    el.addEventListener('pointermove', function (e) {
        if (!el.hasPointerCapture(e.pointerId)) return;
        el.style.left = Math.min(Math.max(0, e.clientX - offsetX), window.innerWidth - el.offsetWidth) + 'px';
        el.style.top = Math.min(Math.max(0, e.clientY - offsetY), window.innerHeight - el.offsetHeight) + 'px';
        el.style.right = 'auto';
        el.style.bottom = 'auto';
    });
    function endDrag(e) {
        if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
        el.style.cursor = 'grab';
    }
    el.addEventListener('pointerup', function (e) {
        endDrag(e);
        if (onClick && Math.hypot(e.clientX - startX, e.clientY - startY) < 4) onClick();
    });
    el.addEventListener('pointercancel', endDrag);
}

// Function to update the legend
function updateLegend() {
    const dropdown = document.getElementById('select2dSurface');
    const selectedOptions = Array.from(dropdown.selectedOptions);
    const legend = document.getElementById('legend');

    // Clear the current legend
    legend.innerHTML = '';

    // Loop over selected options and add to legend
    let selectionIndex = -1;
    selectedOptions.forEach(option => {
        selectionIndex += 1;
        const selectedValue = option.value;
        const color = auxDefaultColor[selectionIndex];  // Get the color for the selected option

        // Create a new legend item
        const legendItem = document.createElement('div');
        legendItem.style.display = 'flex';
        legendItem.style.alignItems = 'center';
        legendItem.style.marginBottom = '5px';

        // Create a colored box for the legend
        const colorBox = document.createElement('div');
        colorBox.style.width = '20px';
        colorBox.style.height = '20px';
        colorBox.style.backgroundColor = color.toCssColorString();  // Assuming auxColor is a Cesium Color object
        colorBox.style.marginRight = '10px';

        // Create a label for the legend item
        const label = document.createElement('span');
        label.textContent = auxLabel[selectedValue];  // Label from the auxLabel array

        // Add the color box and label to the legend item
        legendItem.appendChild(colorBox);
        legendItem.appendChild(label);

        // Add the legend item to the legend div
        legend.appendChild(legendItem);
    });
}

async function loadAuxConfig() {
    try {
        console.log("loadAuxConfig: started");
        const response = await fetch("/load-aux-config-s3/");
        const data = await response.json();

        auxData = data.auxData;
        auxLabel = data.auxLabel;
        auxTitle = data.auxTitle;
        auxCitation = data.auxCitation;
        auxFillOpacity = data.auxFillOpacity;
        auxLineWidth = data.auxLineWidth;
        auxDecimation = data.auxDecimation;
        auxMarker = data.auxMarker;

        console.log("loadAuxConfig: done", auxLabel);
    } catch (error) {
        console.error("Error in loadAuxConfig:", error);
    }
}

function populateDropdown() {
    console.log("populateDropdown called");
    if (!Array.isArray(auxLabel)) {
        console.warn(`auxLabel is not an array, it is a ${typeof auxLabel}.  `, auxLabel);
        return;
    }

    const dropdown = document.getElementById('select2dSurface');
    dropdown.innerHTML = '';

    auxLabel.forEach((label, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = label;
        dropdown.appendChild(option);
    });
}

