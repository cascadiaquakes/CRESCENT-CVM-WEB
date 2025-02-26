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

        fetch('../static/json/' + jsonFlename)
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