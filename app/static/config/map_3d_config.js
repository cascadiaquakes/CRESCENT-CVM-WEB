// CVM area display parameters.
const cvmAreaOutlineColor = [Cesium.Color.CYAN]
const cvmAreaFaceColor = [Cesium.Color.CYAN]
const cvmAreaLabel = ["CVM & model Study Areas"]
const cvmAreaFillOpacity = [0.1]

// Configuration.
const apiKey = "AAPKafd67a0544f04817b08c2f65379b76c8pz3w8RSH_npDJjf9phbqEJ2kbD8QnfX-lzVlJ7dUi_3pQjwWS-vNFeXT6jacicfJ";
Cesium.Ion.defaultAccessToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJmZDQzMzIyOS1lZDFlLTRhNTgtYTE1Yy04YzNkNWQ2ZmI4OTIiLCJpZCI6MjAxODU5LCJpYXQiOjE3MTA0MDQ5MDd9._E6MFZMMjbxpzC4qYYROP1ldtV1MJn0f56W5woAtboc";

// Initial view settings
const initialFlyTO = [-132.76, 39.84, 718005]// Increase the altitude value to zoom out
const initialHeading = 48.65
const initialPitch = -36.05
const initialRoll = 0.3

// Earthquakes
const eqQueryUrl = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=1970-01-01&minmagnitude=4&minlatitude=' + south + '&maxlatitude=' + north + '&minlongitude=' + west + '&maxlongitude=' + east;
const eqLabel = ['Earthquakes (M ≥ 4), Circle Size ∝ Magnitude'];
const eqColor = Cesium.Color.YELLOW;
const eqMarker = ["circle"];

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


var dataSources = [];
const auxData = ["https://raw.githubusercontent.com/cascadiaquakes/crescent-cfm/main/crescent_cfm_files/cascadia_subduction_interface_temp.geojson", "/static/geojson/Hayesetal2018_slab2_v2018.geojson", "/static/geojson/McCroryetal_gmtcontour.geojson", "/static/geojson/Delphetal-2021.geojson"];
const auxLabel = ["CFM - Cascadia Subduction Interface", "Hayes et al., 2018 Slab2", "McCrory et al., 2012 Juan de Fuca slab", "Delph et al., 2021 Modified McCrory et al 2012"];
const auxColor = [Cesium.Color.GREEN, Cesium.Color.ORANGE, Cesium.Color.PINK, Cesium.Color.YELLOW];
const auxFillOpacity = [0.3, 0.3, 0.3, 0.3];
const auxLineWidth = [4, 4, 4, 4];
const auxMarker = ["rectangle", "rectangle", "rectangle", "rectangle"];

/*
var data = ['https://raw.githubusercontent.com/cascadiaquakes/crescent-cfm/main/crescent_cfm_files/crescent_cfm_crustal_traces.geojson', 'https://raw.githubusercontent.com/cascadiaquakes/crescent-cfm/main/crescent_cfm_files/crescent_cfm_crustal_3d.geojson'];
var label = ['CRESCENT Cascadia Fault Traces', 'CRESCENT Cascadia Fault Surfaces', 'Earthquakes (M ≥ 4), Circle Size ∝ Magnitude'];
//const geojson_files = ['https://raw.githubusercontent.com/cascadiaquakes/crescent-cfm/main/crescent_cfm_files/crescent_cfm_crustal_traces.geojson', 'https://raw.githubusercontent.com/cascadiaquakes/crescent-cfm/main/crescent_cfm_files/crescent_cfm_crustal_3d.geojson']
const dataColor = [Cesium.Color.CRIMSON, Cesium.Color.DARKMAGENTA, Cesium.Color.YELLOW];
const fillOpacity = [0.5, 0.5];
const lineWidth = [2, 2];
*/

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

var data = cfmData.concat(cfmTraceData)
var label = cfmLabel.concat(cfmTraceLabel)
const dataColor = cfmColor.concat(cfmTraceColor)
const fillOpacity = cfmFillOpacity.concat(cfmFillOpacity)
const lineWidth = cfmLineWidth.concat(cfmTraceLineWidth);
const marker = cfmMarker.concat(cfmTraceMarker)

// Legends
var primaryLegendLabel = cfmLabel.concat(cfmTraceLabel);
var primaryLegendColor = cfmColor.concat(cfmTraceColor);
var auxLegendLabel = cvmAreaLabel.concat(cvmLabel).concat(eqLabel).concat(boundaryLabel);
var auxLegendColor = cvmAreaOutlineColor.concat(cvmColor).concat(eqColor).concat(boundaryColor);