# Political boundaries (3D viewer)

Both files come from Natural Earth 1:10m Admin 1 states and provinces (public domain), `ne_10m_admin_1_states_provinces`, filtered to `adm0_a3` USA and CAN.

They are stored as MultiLineString outlines, not polygons. Cesium triangulates polygons even with a transparent fill, and at this resolution that exhausted browser memory. Lines render the same outline far more cheaply.

Detail is set per island or coastline piece. Pieces inside lon -135 to -110, lat 35 to 56 (Cascadia plus a margin) are simplified at 0.005 degrees, about 3.5 km median segment length. Pieces outside are only drawn faded, so they are simplified at 0.05 degrees and islands under 0.5 square degrees are dropped. Coordinates are rounded to 4 decimal places and only the `name` property is kept.
