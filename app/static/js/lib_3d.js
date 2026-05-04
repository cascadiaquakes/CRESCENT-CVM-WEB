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

