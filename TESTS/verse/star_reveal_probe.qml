import QtQuick
import "../../PRIMORA/verse/qml" as VerseQml

// Test harness only: drives a real Starfield through reveal values and
// records per-star gates, so Python tests stay deterministic without
// wall-clock timing or pixel reads.
Item {
    id: root
    property var probe: []
    property var phases: []
    property var reveals: []

    VerseQml.Starfield {
        id: sf
    }

    function run() {
        var out = []
        for (var i = 0; i < reveals.length; i++) {
            sf.reveal = reveals[i]
            var row = []
            for (var j = 0; j < phases.length; j++) {
                row.push(sf.gateFor(phases[j]))
            }
            out.push(row)
        }
        probe = out
    }
}
