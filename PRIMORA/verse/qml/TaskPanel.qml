import QtQuick
import QtQuick.Controls

// V1 task/execution area: a small status indicator that reveals the
// compact panel on hover (or pins it open with a click). The Universe
// stays dominant; the full panel is never permanently visible.
Item {
    id: taskPanel
    property var verseApp: null
    property bool pinned: false
    readonly property bool expanded: pinned || hoverArea.containsMouse

    // Connect to verseApp task state changes
    // We handle the target binding manually since verseApp may be null initially
    Connections {
        id: verseAppConnections
        target: verseApp
        function onTaskStateChanged(state) {
            updateDisplay(state)
        }
    }

    Component.onCompleted: {
        if (verseApp) {
            verseAppConnections.target = verseApp
        }
    }

    onVerseAppChanged: {
        if (verseApp) {
            verseAppConnections.target = verseApp
        }
    }

    Rectangle {
        id: panel
        anchors.fill: parent
        visible: taskPanel.expanded
        color: "#081018"
        opacity: 0.94
        border.color: "#2a3a4a"
        border.width: 1
        radius: 8

        Column {
            anchors.fill: parent
            anchors.margins: 12
            spacing: 8

            Row {
                spacing: 8
                Text {
                    text: "TASK"
                    color: "#6a8ab0"
                    font.family: "Segoe UI"
                    font.pixelSize: 11
                    font.bold: true
                    font.letterSpacing: 1.5
                }
                Rectangle {
                    id: statusDot
                    width: 8
                    height: 8
                    radius: 4
                    color: taskPanel._statusColor
                }
            }

            Text {
                id: descriptionText
                text: taskPanel._description
                color: "#a0b0c0"
                font.family: "Segoe UI"
                font.pixelSize: 13
                wrapMode: Text.WordWrap
                width: parent.width
            }

            Text {
                id: resultText
                text: taskPanel._result
                color: taskPanel._resultColor
                font.family: "Segoe UI"
                font.pixelSize: 12
                wrapMode: Text.WordWrap
                width: parent.width
                visible: taskPanel._result.length > 0
            }
        }
    }

    // Collapsed state: a single subtle status dot, top-right.
    Item {
        id: indicator
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 8
        width: 20
        height: 20
        visible: !taskPanel.expanded

        Rectangle {
            anchors.fill: parent
            radius: 10
            color: "#00000000"
            border.color: taskPanel._statusColor
            border.width: 1
            opacity: 0.5
        }
        Rectangle {
            anchors.centerIn: parent
            width: 8
            height: 8
            radius: 4
            color: taskPanel._statusColor
            opacity: 0.9
        }
    }

    // Hover reveals, click pins open (touch / precise control).
    MouseArea {
        id: hoverArea
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: taskPanel.pinned = !taskPanel.pinned
    }

    property string _statusColor: "#6a8ab0"
    property string _description: "No active task"
    property string _result: ""
    property string _resultColor: "#a0b0c0"

    function updateDisplay(state) {
        if (!state) return

        switch (state.status) {
            case "idle":
                _statusColor = "#6a8ab0"
                _description = "No active task"
                _result = ""
                break
            case "running":
                _statusColor = "#b0a040"
                _description = state.description || "Task running..."
                _result = ""
                break
            case "completed":
                _statusColor = "#4ab060"
                _description = state.description || "Task completed"
                _result = state.result || "Done"
                _resultColor = "#4ab060"
                break
            case "failed":
                _statusColor = "#b04040"
                _description = state.description || "Task failed"
                _result = state.result || "Error"
                _resultColor = "#b04040"
                break
        }
    }
}