import QtQuick

// Minimal reusable placeholder presentation for the M1 shell.
// Deliberately empty of Universe visuals (stars, rays, animation belong
// to later milestones). Easy to replace in M2.
Item {
    id: placeholder

    Text {
        anchors.centerIn: parent
        text: "PRIMORA Verse"
        color: "#8b95a7"
        font.pixelSize: 28
    }

    Text {
        anchors.top: parent.verticalCenter
        anchors.topMargin: 28
        anchors.horizontalCenter: parent.horizontalCenter
        text: "placeholder shell"
        color: "#4a5568"
        font.pixelSize: 14
    }
}
