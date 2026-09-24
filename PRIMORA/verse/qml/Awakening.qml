import QtQuick

// Failure readout only. The awakening experience itself is the dormant
// Universe becoming alive (see Universe.dormancy); this item carries no
// loading text, spinners, progress, or animation of any kind.
Item {
    id: awakening
    property string errorText: ""
    visible: errorText !== ""

    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 28
        width: Math.min(parent.width * 0.7, 560)
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.WordWrap
        text: parent.errorText
        color: "#9a7a72"
        font.family: "Segoe UI"
        font.pixelSize: 13
    }
}
