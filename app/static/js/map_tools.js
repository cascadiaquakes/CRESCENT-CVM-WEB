// Shared navigation tools for the 2D and 3D Cesium viewers, matching the
// earthquake catalog viewer: zoom buttons, keyboard navigation, a scale bar,
// and a cursor coordinate readout.
(function () {
    'use strict';

    function cameraHeight(viewer) {
        return Math.max(Math.abs(viewer.camera.positionCartographic.height), 1000);
    }

    // Farthest the camera may zoom out, in meters (set by init's maxZoomOut).
    let maxHeight = Infinity;

    function zoom(viewer, direction) {
        const height = cameraHeight(viewer);
        let amount = height * 0.35;
        if (direction < 0) amount = Math.min(amount, Math.max(0, maxHeight - height));
        if (direction > 0) viewer.camera.zoomIn(amount); else if (amount > 0) viewer.camera.zoomOut(amount);
    }

    function addZoomButtons(viewer, bottom, right) {
        const group = document.createElement('div');
        group.className = 'map-zoom-group';
        group.style.bottom = bottom;
        group.style.right = right;
        [['+', 'Zoom in (+ key)', 1], ['−', 'Zoom out (− key)', -1]].forEach(function (b) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.textContent = b[0];
            btn.title = b[1];
            btn.setAttribute('aria-label', b[1]);
            btn.addEventListener('click', function () { zoom(viewer, b[2]); });
            group.appendChild(btn);
        });
        document.body.appendChild(group);
    }

    function addKeyboardNavigation(viewer) {
        document.addEventListener('keydown', function (e) {
            const tag = (e.target && e.target.tagName) || '';
            if (/INPUT|TEXTAREA|SELECT/.test(tag) || e.target.isContentEditable || e.ctrlKey || e.metaKey || e.altKey) return;
            const step = cameraHeight(viewer) * 0.08;
            const cam = viewer.camera;
            switch (e.key) {
                case '+': case '=': zoom(viewer, 1); break;
                case '-': case '_': zoom(viewer, -1); break;
                case 'ArrowUp': cam.moveUp(step); break;
                case 'ArrowDown': cam.moveDown(step); break;
                case 'ArrowLeft': cam.moveLeft(step); break;
                case 'ArrowRight': cam.moveRight(step); break;
                default: return;
            }
            e.preventDefault();
        });
    }

    // Add the keyboard shortcuts to the mouse tab of Cesium's "?" help panel.
    function addKeyboardHelp(viewer) {
        const table = viewer.container.querySelector('.cesium-navigation-help-instructions table');
        if (!table) return;
        const row = document.createElement('tr');
        row.innerHTML = '<td class="map-keyboard-help__icon" aria-hidden="true">&#9000;</td>' +
            '<td><div class="map-keyboard-help__title">Keyboard</div>' +
            '<div class="cesium-navigation-help-details">+ / &minus; to zoom</div>' +
            '<div class="cesium-navigation-help-details">Arrow keys to pan</div></td>';
        (table.tBodies[0] || table).appendChild(row);
    }

    function niceDistance(maxMeters) {
        const pow = Math.pow(10, Math.floor(Math.log10(maxMeters)));
        const steps = [5, 2, 1];
        for (let i = 0; i < steps.length; i++) {
            if (steps[i] * pow <= maxMeters) return steps[i] * pow;
        }
        return pow;
    }

    function addScaleBar(viewer, position) {
        const MAX_PX = 120;
        const el = document.createElement('div');
        el.className = 'map-scale-bar map-scale-bar--' + (position || 'bottom-left');
        el.innerHTML = '<div class="map-scale-bar__label"></div><div class="map-scale-bar__line"></div>';
        document.body.appendChild(el);
        const label = el.querySelector('.map-scale-bar__label');
        const line = el.querySelector('.map-scale-bar__line');
        const geodesic = new Cesium.EllipsoidGeodesic();
        let last = 0;

        viewer.scene.postRender.addEventListener(function () {
            const now = performance.now();
            if (now - last < 200) return;
            last = now;
            const canvas = viewer.scene.canvas;
            const y = canvas.clientHeight - 60;
            const x = canvas.clientWidth / 2;
            const left = viewer.camera.pickEllipsoid(new Cesium.Cartesian2(x, y));
            const right = viewer.camera.pickEllipsoid(new Cesium.Cartesian2(x + 1, y));
            if (!left || !right) { el.style.display = 'none'; return; }
            const ell = viewer.scene.globe.ellipsoid;
            geodesic.setEndPoints(ell.cartesianToCartographic(left), ell.cartesianToCartographic(right));
            const metersPerPixel = geodesic.surfaceDistance;
            if (!Number.isFinite(metersPerPixel) || metersPerPixel <= 0) { el.style.display = 'none'; return; }
            const meters = niceDistance(metersPerPixel * MAX_PX);
            line.style.width = Math.round(meters / metersPerPixel) + 'px';
            label.textContent = meters >= 1000 ? (meters / 1000) + ' km' : meters + ' m';
            el.style.display = '';
        });
    }

    function addCoordinateReadout(viewer, bottom) {
        const el = document.createElement('div');
        el.className = 'map-coords';
        el.style.bottom = bottom;
        el.innerHTML = '<span class="map-coords__lat">--</span><span class="map-coords__sep">,</span><span class="map-coords__lon">--</span>';
        document.body.appendChild(el);
        const lat = el.querySelector('.map-coords__lat');
        const lon = el.querySelector('.map-coords__lon');
        const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
        handler.setInputAction(function (movement) {
            const cartesian = viewer.camera.pickEllipsoid(movement.endPosition);
            if (!cartesian) { el.style.opacity = '0.5'; return; }
            const c = Cesium.Cartographic.fromCartesian(cartesian);
            lat.textContent = Cesium.Math.toDegrees(c.latitude).toFixed(3);
            lon.textContent = Cesium.Math.toDegrees(c.longitude).toFixed(3);
            el.style.opacity = '1';
        }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
    }

    function addStyles() {
        if (document.getElementById('cfm-map-tools-style')) return;
        const css = `
            .map-zoom-group { position: fixed; z-index: 1300; display: flex; flex-direction: column;
                background: #fff; border-radius: 6px; box-shadow: 0 0 0 2px rgba(0,0,0,0.1); overflow: hidden; }
            .map-zoom-group button { width: 32px; height: 32px; border: 0; background: #fff; color: #1e293b;
                font: 600 20px/1 -apple-system, 'Segoe UI', Roboto, sans-serif; cursor: pointer; padding: 0; }
            .map-zoom-group button + button { border-top: 1px solid #e2e8f0; }
            .map-zoom-group button:hover { background: #f1f5f9; }
            .map-scale-bar { position: fixed; z-index: 400; pointer-events: none;
                font: 600 11px Inter, -apple-system, 'Segoe UI', Roboto, sans-serif; color: #1e293b;
                background: rgba(255, 255, 255, 0.85); border-radius: 6px; padding: 3px 6px 4px; }
            .map-scale-bar--bottom-left { left: 12px; bottom: 44px; }
            .map-scale-bar--bottom-right { right: 12px; bottom: 34px; }
            .map-scale-bar__label { text-align: left; margin-bottom: 2px; }
            .map-scale-bar__line { height: 6px; border: 2px solid #1e293b; border-top: none; }
            .map-coords { position: fixed; left: 50%; transform: translateX(-50%); z-index: 1200;
                pointer-events: none; user-select: none; display: inline-flex; align-items: center; gap: 6px;
                background: rgba(0, 0, 0, 0.35); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 14px; padding: 6px 10px;
                font: 600 12px Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
                color: rgba(255, 255, 255, 0.92); box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18); }
            .map-coords__sep { color: rgba(255, 255, 255, 0.7); }
            .map-keyboard-help__icon { font-size: 26px; text-align: center; color: #fff; padding: 0 4px; }
            .map-keyboard-help__title { color: #f472b6; font-weight: bold; }
        `;
        const style = document.createElement('style');
        style.id = 'cfm-map-tools-style';
        style.textContent = css;
        document.head.appendChild(style);
    }

    window.CfmMapTools = {
        // options: zoomBottom / zoomRight (CSS, fixed to the window), scalePosition, coordsBottom,
        // coords (false when the page has its own readout), maxZoomOut (meters, keeps the camera regional)
        init: function (viewer, options) {
            options = options || {};
            if (options.maxZoomOut) {
                maxHeight = options.maxZoomOut;
                viewer.scene.screenSpaceCameraController.maximumZoomDistance = options.maxZoomOut;
            }
            addStyles();
            addZoomButtons(viewer, options.zoomBottom || '90px', options.zoomRight || '12px');
            addKeyboardNavigation(viewer);
            addKeyboardHelp(viewer);
            addScaleBar(viewer, options.scalePosition);
            if (options.coords !== false) addCoordinateReadout(viewer, options.coordsBottom || '14px');
        }
    };
})();
