// When a viewer template is loaded inside the main map's tool panel iframe,
// flag the document so the embedded-scoped CSS rules apply (they constrain
// #dynamicPlot to max-width:100% so the returned PNG fits the panel width
// instead of overflowing to the right at its natural size).
(function () {
    try {
        if (window.parent && window.parent !== window) {
            document.documentElement.classList.add('embedded');
        }
    } catch (e) { /* cross-origin; ignore */ }
})();

/**
       * Generates HTML content from a given JSON object, handling nested structures recursively.
       *
       * @param {Object} jsonData - The JSON object to be converted into HTML.
       * @returns {string} - The generated HTML content as a string.
       */
function createHtmlFromJson(jsonData) {
    /**
     * Recursively processes JSON data to generate HTML content.
     * 
     * @param {Object} jsonData - The current level of the JSON object to be processed.
     * @returns {string} - The HTML content for the current level of the JSON object.
     */
    function processJson(jsonData, counter) {
        // Add the hide checkbox only at the top.
        let htmlContent = '<div class="json-container"><div>'
        if (counter == 0) {
            htmlContent += '<input type="checkbox" id="hideCheckbox" onchange="document.getElementById(\'jsonContent\').style.display=\'none\';" ><label for="hideCheckbox" style="font-weight:bold;color:maroon;"> Hide Metadata</label ><hr /></div> ';
        }
        for (let key in jsonData) {
            if (typeof jsonData[key] === 'object' && jsonData[key] !== null) {
                htmlContent += `<div><b>${key}:</b>${processJson(jsonData[key], counter + 1)}</div>`;
            } else {
                htmlContent += `<div><b>${key}:</b> ${jsonData[key]}</div>`;
            }
        }
        htmlContent += '</div>';
        return htmlContent;
    }

    return processJson(jsonData, 0);
}

// Display file metadata.
function display_metadata() {
    const model = document.getElementById('data-file');

    if (model.value) {
        const regex = /.nc/i;
        const jsonFlename = model.value.replace(regex, '.json')
        // Field is found, execute your function
        // Construct the API URL
        const apiUrl = `/data/fetch_json_s3?file_name=${encodeURIComponent(jsonFlename)}`;
        fetch(apiUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(jsonData => {
                // Display JSON content in the panel
                // document.getElementById("mmodelname").innerHTML = " Metadata";
                jsonContent = document.getElementById("jsonContent")
                jsonContent.innerHTML = createHtmlFromJson(jsonData);
                jsonContent.style.display = 'block'
            })
            .catch(error => {
                console.error('There was a problem with the fetch operation:', error);
            });
    }
}

// Toggle description.
function toggleDescription(title) {
    var description = document.getElementById("description");
    //var button = document.getElementById("toggleDescription");
    if (description.style.display === "none") {
        description.style.display = "block";
        //button.innerHTML = "&#9650;&nbsp;" + title; // Up arrow
    } else {
        description.style.display = "none";
        //button.innerHTML = "&#9660;&nbsp;" + title; // Down arrow
    }
}


//Large file animation/extraction message
function showLoadingMessage(options = {}) {
    const {
        size_kb,
        action = "processing",
        threshold = 10000.0,
        containerId = "loadingMessage",
        notesId = "notes",
        largeColor = "red",
        normalColor = "black",
        fontSize = "11px"
    } = options;

    const isLarge = size_kb !== undefined && size_kb >= threshold;
    const color = isLarge ? largeColor : normalColor;

    const mainMessage = isLarge
        ? `Large file – ${action} may take time`
        : `${action.charAt(0).toUpperCase() + action.slice(1)}&nbsp;`;

    const blinkingSpan = `&nbsp;<span class="blinking-dots">...</span>working`;
    const loadingHTML = `<b style="color:${color};font-size:${fontSize};">${mainMessage}${blinkingSpan}</b>`;

    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = loadingHTML;
    } else {
        console.warn(`Element with ID '${containerId}' not found.`);
    }

    const notesElem = document.getElementById(notesId);
    if (notesElem) {
        notesElem.innerHTML = isLarge
            ? `<span style="color:${largeColor};font-size:${fontSize};">Note: Large file – ${action} may take time to complete.</span>`
            : "";
    }

    // Inject blinking-dots style once
    if (!document.getElementById("blinking-style")) {
        const style = document.createElement("style");
        style.id = "blinking-style";
        style.innerHTML = `
            .blinking-dots {
                animation: blink 1s steps(1, start) infinite;
            }
            @keyframes blink {
                50% { opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}


// -------------------------------------------------------------
// Color-scale control on the rendered plot (eqcat-style gear +
// popup with Auto vs Custom Min/Max, re-renders via the same
// form submit path). Used by depth-slice + cross-section viewers.
// -------------------------------------------------------------
function initPlotColorRangeControl(opts) {
    opts = opts || {};
    const formId = opts.formId || 'image-form';
    const plotId = opts.plotId || 'dynamicPlot';
    const minInputName = opts.minInputName || 'start_value';
    const maxInputName = opts.maxInputName || 'end_value';

    const plot = document.getElementById(plotId);
    const form = document.getElementById(formId);
    if (!plot || !form) return;
    if (document.getElementById('colorRangeToolbar')) return;

    const toolbar = document.createElement('div');
    toolbar.id = 'colorRangeToolbar';
    toolbar.style.cssText = 'display:none;position:relative;margin:8px 0 12px;';

    const gearBtn = document.createElement('button');
    gearBtn.type = 'button';
    gearBtn.id = 'colorRangeGear';
    gearBtn.title = 'Color scale settings';
    gearBtn.style.cssText = 'display:inline-flex;align-items:center;gap:6px;padding:6px 12px;background:#f1f5f9;color:#0b4a53;border:1px solid #cbd5e1;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;';
    gearBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 15.5A3.5 3.5 0 0 1 8.5 12A3.5 3.5 0 0 1 12 8.5a3.5 3.5 0 0 1 3.5 3.5a3.5 3.5 0 0 1-3.5 3.5m7.4-2.5a7 7 0 0 0 0-2l2.1-1.6c.2-.2.2-.4.1-.6l-2-3.5c-.1-.2-.4-.3-.6-.2l-2.5 1a7 7 0 0 0-1.7-1L14 2.4a.5.5 0 0 0-.5-.4h-4a.5.5 0 0 0-.5.4l-.4 2.7a7 7 0 0 0-1.7 1l-2.5-1c-.2-.1-.5 0-.6.2l-2 3.5c-.1.2-.1.4.1.6L4 11a7 7 0 0 0 0 2L2 14.6c-.2.2-.2.4-.1.6l2 3.5c.1.2.4.3.6.2l2.5-1c.5.4 1 .7 1.7 1l.4 2.7c0 .2.2.4.5.4h4c.3 0 .5-.2.5-.4l.4-2.7a7 7 0 0 0 1.7-1l2.5 1c.2.1.5 0 .6-.2l2-3.5c.1-.2.1-.4-.1-.6l-2.1-1.6Z"/></svg>Color scale';
    toolbar.appendChild(gearBtn);

    const popup = document.createElement('div');
    popup.id = 'colorRangePopup';
    popup.style.cssText = 'display:none;position:absolute;top:calc(100% + 6px);left:0;z-index:20;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,0.12);padding:14px;min-width:260px;';
    popup.innerHTML = [
        '<div style="font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;color:#64748b;margin-bottom:10px;">Color scale settings</div>',
        '<label style="display:flex;gap:8px;align-items:center;font-size:12px;color:#334155;margin-bottom:6px;cursor:pointer;">',
        '  <input type="radio" name="colorMode" value="auto" checked>Auto-adjust to data',
        '</label>',
        '<label style="display:flex;gap:8px;align-items:center;font-size:12px;color:#334155;cursor:pointer;">',
        '  <input type="radio" name="colorMode" value="custom">Custom range',
        '</label>',
        '<div id="colorCustomInputs" style="display:none;margin-top:10px;">',
        '  <div style="display:flex;gap:10px;">',
        '    <label style="flex:1;font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;">',
        '      <span id="colorMinLabel">Min</span>',
        '      <input type="number" id="colorMin" step="any" style="width:100%;box-sizing:border-box;padding:6px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:12px;margin-top:4px;">',
        '    </label>',
        '    <label style="flex:1;font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;">',
        '      <span id="colorMaxLabel">Max</span>',
        '      <input type="number" id="colorMax" step="any" style="width:100%;box-sizing:border-box;padding:6px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:12px;margin-top:4px;">',
        '    </label>',
        '  </div>',
        '</div>',
        '<div style="display:flex;gap:8px;margin-top:14px;">',
        '  <button type="button" id="colorRangeReset" style="flex:1;padding:7px 10px;background:#f1f5f9;color:#64748b;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;">Reset to Auto</button>',
        '  <button type="button" id="colorRangeApply" style="flex:1;padding:7px 10px;background:#0b4a53;color:#ffffff;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;">Apply</button>',
        '</div>'
    ].join('');
    toolbar.appendChild(popup);

    plot.parentNode.insertBefore(toolbar, plot);

    const customInputs = document.getElementById('colorCustomInputs');
    const minInput = document.getElementById('colorMin');
    const maxInput = document.getElementById('colorMax');

    // Unit lookup for the common CVM plot variables. Anything unmapped
    // shows the raw variable name without a unit annotation.
    const _unitFor = function (v) {
        if (!v) return '';
        const key = String(v).toLowerCase();
        if (key === 'vs' || key === 'vp') return 'km/s';
        if (key === 'density' || key === 'rho') return 'g/cm³';
        if (key.startsWith('vs_uncert') || key === 'uncert' || key.includes('uncertainty')) return 'km/s';
        return '';
    };
    const _refreshRangeLabels = function () {
        const varEl = document.getElementById('plot-variable') || form.querySelector('[name="plot_variable"]');
        const raw = varEl ? (varEl.value || '') : '';
        const label = raw.charAt(0).toUpperCase() + raw.slice(1);
        const unit = _unitFor(raw);
        const suffix = label ? (' ' + label + (unit ? ' (' + unit + ')' : '')) : '';
        const minL = document.getElementById('colorMinLabel');
        const maxL = document.getElementById('colorMaxLabel');
        if (minL) minL.textContent = 'Min' + suffix;
        if (maxL) maxL.textContent = 'Max' + suffix;
    };

    gearBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        _refreshRangeLabels();
        popup.style.display = popup.style.display === 'block' ? 'none' : 'block';
    });
    document.addEventListener('click', function (e) {
        if (!toolbar.contains(e.target)) popup.style.display = 'none';
    });
    document.querySelectorAll('input[name="colorMode"]').forEach(function (r) {
        r.addEventListener('change', function () {
            const mode = document.querySelector('input[name="colorMode"]:checked').value;
            customInputs.style.display = mode === 'custom' ? 'block' : 'none';
        });
    });
    document.getElementById('colorRangeReset').addEventListener('click', function () {
        document.querySelector('input[name="colorMode"][value="auto"]').checked = true;
        customInputs.style.display = 'none';
        minInput.value = '';
        maxInput.value = '';
        const minEl = form.querySelector('[name="' + minInputName + '"]');
        const maxEl = form.querySelector('[name="' + maxInputName + '"]');
        if (minEl) minEl.value = 'auto';
        if (maxEl) maxEl.value = 'auto';
        popup.style.display = 'none';
        form.requestSubmit();
    });
    document.getElementById('colorRangeApply').addEventListener('click', function () {
        const mode = document.querySelector('input[name="colorMode"]:checked').value;
        const minEl = form.querySelector('[name="' + minInputName + '"]');
        const maxEl = form.querySelector('[name="' + maxInputName + '"]');
        if (mode === 'auto') {
            if (minEl) minEl.value = 'auto';
            if (maxEl) maxEl.value = 'auto';
        } else {
            if (!minInput.value || !maxInput.value) {
                alert('Enter both Min and Max for the custom range.');
                return;
            }
            if (Number(minInput.value) >= Number(maxInput.value)) {
                alert('Min must be less than Max.');
                return;
            }
            if (minEl) minEl.value = minInput.value;
            if (maxEl) maxEl.value = maxInput.value;
        }
        popup.style.display = 'none';
        form.requestSubmit();
    });
}

function showColorRangeControl() {
    const t = document.getElementById('colorRangeToolbar');
    if (t) t.style.display = 'block';
}