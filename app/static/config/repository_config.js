// CVM area display parameters.
const cvmAreaOutlineColor = [Cesium.Color.CYAN];
const cvmAreaFaceColor = [Cesium.Color.CYAN];
const cvmAreaLabel = ["CVM & Study (—) Areas; Filters (- -)"];
const cvmAreaFillOpacity = [0.1];

// Configuration.
Cesium.Ion.defaultAccessToken = "your_access_token";

// Initial view settings
const initialFlyTO = [-132.76, 39.84, 1500000]; // Increase the altitude value to zoom out
const initialHeading = 48.65;
const initialPitch = -75;
const initialRoll = 0.3;

// Earthquakes
const eqQueryUrl = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=1970-01-01&minmagnitude=4&minlatitude=' + south + '&maxlatitude=' + north + '&minlongitude=' + west + '&maxlongitude=' + east;
const eqLabel = ['Earthquakes (M ≥ 4), Circle Size ∝ Magnitude'];
const eqColor = Cesium.Color.YELLOW;
const eqMarker = ["circle"];

// NOTE: if you change any of the following colors, you must change the corresponding selection box rgb color in the index file under modelSelectionPanel.
const cesiumColorList = [
    Cesium.Color.CYAN,
    Cesium.Color.RED,
    Cesium.Color.GREEN,
    Cesium.Color.BLUE,
    Cesium.Color.YELLOW,
    Cesium.Color.WHITE,
    Cesium.Color.BROWN,
    Cesium.Color.PURPLE,
    Cesium.Color.PINK
];

// CVM display parameters.
const cvmOutlineColor = Cesium.Color.BLUE;
const cvmFaceColor = Cesium.Color.BLUE;
const cvmLabel = ["CVM Coverage"];
const cvmColor = [cvmFaceColor];
const cvmMarker = ["rectangle"];
const cvmLineWidth = 4;


// Auxiliary data.
const auxData = ["https://raw.githubusercontent.com/cascadiaquakes/crescent-cfm/main/crescent_cfm_files/cascadia_subduction_interface_temp.geojson",
    "/static/geojson/Hayesetal2018_slab2_v2018.geojson",
    "/static/geojson/McCroryetal_gmtcontour.geojson",
    "/static/geojson/Delphetal-2021.geojson",
    "/static/geojson/Casie21-R2T-TOC_medflt-surface-mask_d2.geojson",
    "/static/geojson/Casie21-R2T-PlateBdy_medflt-surface-mask_d2.geojson"];

const auxLabel = ["Graham et al., 2018 - Cascadia Subduction Interface",
    "Hayes et al., 2018 - Slab2",
    "McCrory et al., 2012 - Juan de Fuca slab",
    "Delph et al., 2021 - Modified McCrory et al 2012",
    "Carbotte  et al., 2024 - Top of igneous Oceanic Crust",
    "Carbotte  et al., 2024 - Plate boundary/megathrust fault"];
const auxTitle = ["Cascadia Subduction Interface",
    "A Comprehensive Subduction Zone Geometry Model, Cascadia",
    "JdF slab geometry beneath the Cascadia subduction margin",
    "Delph2021 slab model made from modifying McCrory et al 2012 slab",
    "Depth to the Top of igneous Oceanic Crust (TOC)",
    "Depth to the interpreted plate boundary/megathrust fault"
];
const auxCitation = [["Graham, S. E., Loveless, J. P., & Meade, B. J., 2018", "https://doi.org/10.1029/2017GC007391"],
["Hayes, G., 2018", "https://doi.org/10.5066/F7PV6JNV"],
["McCrory et al., 2012", "https://doi.org/10.1029/2012JB009407"],
["Delph, Thomas, Levander, 2021", "https://doi.org/10.1016/j.epsl.2020.116724"],
["Carbotte et al., Science Advances, 2024", "https://doi.org/10.1126/sciadv.adl3198"],
["Carbotte et al., Science Advances, 2024", "https://doi.org/10.1126/sciadv.adl3198"]
];
const auxDefaultColor = [Cesium.Color.OLIVE.withAlpha(0.7), Cesium.Color.SIENNA.withAlpha(0.7)]
const auxFillOpacity = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
const auxLineWidth = [4, 4, 4, 4, 4, 4];
const auxDecimation = [1, 1, 1, 1, 2, 2];
const auxMarker = ["rectangle", "rectangle", "rectangle", "rectangle", "rectangle", "rectangle"];

// Boundary data.
const boundaryData = [
    '/static/boundary_geojson/us-states.json',
    '/static/boundary_geojson/georef-canada-province-public.geojson'
];
const boundaryLabel = ['US', 'Canada'];
const boundaryColor = [Cesium.Color.GRAY, Cesium.Color.DIMGRAY];
const boundaryFillOpacity = [0.0, 0.0];
const boundaryLineWidth = [2, 0.5];
const boundaryMarker = ["line"];


// CFM data
const cfmData = ['https://raw.githubusercontent.com/cascadiaquakes/crescent-cfm/main/crescent_cfm_files/crescent_cfm_crustal_3d.geojson'];
const cfmLabel = ['Cascadia Fault Surfaces'];
const cfmColor = [Cesium.Color.DARKMAGENTA];
const cfmFillOpacity = [0.5];
const cfmLineWidth = [2];
const cfmMarker = ["rectangle"];

// CFM Trace data
const cfmTraceData = ['https://raw.githubusercontent.com/cascadiaquakes/crescent-cfm/main/crescent_cfm_files/crescent_cfm_crustal_traces.geojson'];
const cfmTraceLabel = ['Cascadia Fault Traces'];
const cfmTraceColor = [Cesium.Color.CRIMSON];
const cfmTraceFillOpacity = [0.5];
const cfmTraceLineWidth = [2];
const cfmTraceMarker = ["line"];

// Map labels to checkboxes on the HTML page.
const dataSourceCheckboxMapping = {
    'Cascadia Fault Traces': 'toggleCFMTraceCheckbox',
    'Cascadia Fault Surfaces': 'toggleCFMCheckbox'
};

// Legends
var primaryLegendLabel = cfmLabel.concat(cfmTraceLabel);
var primaryLegendColor = cfmColor.concat(cfmTraceColor);
var auxLegendLabel = cvmAreaLabel.concat(cvmLabel).concat(eqLabel).concat(boundaryLabel);
var auxLegendColor = cvmAreaOutlineColor.concat(cvmColor).concat(eqColor).concat(boundaryColor);

// Legacy support.
var data = cfmData.concat(cfmTraceData)
var label = cfmLabel.concat(cfmTraceLabel)

const dataColor = cfmColor.concat(cfmTraceColor)
const fillOpacity = cfmFillOpacity.concat(cfmFillOpacity)
const lineWidth = cfmLineWidth.concat(cfmTraceLineWidth);
const marker = cfmMarker.concat(cfmTraceMarker)

// Check if Cesium.Ion.defaultAccessToken is not set or is using the placeholder token.
// using a synchronous XHR request, ensuring that the Cesium token is fetched and 
// set before any further processing. This will block the browser until the token 
// is fetched, but it guarantees that the token is available before proceeding.
// Synchronous XHR is deprecated and will block page rendering, so this method 
// should only be used as a last resort.However, it guarantees that the token 
// is set before any dependent scripts run. Our fetch is quick and is OK to use.
if (!Cesium.Ion.defaultAccessToken || Cesium.Ion.defaultAccessToken === "your_access_token") {
    console.log("Cesium access token not set or is using the placeholder token. Retrieving from server..." + Cesium.Ion.defaultAccessToken);

    try {
        // Create a synchronous XHR request to fetch the token
        const xhr = new XMLHttpRequest();
        xhr.open('GET', '/get-token', false);  // `false` makes the request synchronous
        xhr.send();

        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            Cesium.Ion.defaultAccessToken = data.token;
            console.log("Cesium access token set successfully.");// + Cesium.Ion.defaultAccessToken);
        } else {
            throw new Error("Failed to fetch Cesium access token from server.");
        }
    } catch (error) {
        console.error("Error fetching Cesium access token:", error);
    }
} else {
    console.log("Cesium access token is already set and valid.");
}
