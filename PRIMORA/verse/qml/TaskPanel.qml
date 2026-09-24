import QtQuick
import QtQuick.Controls

// V1 task/execution area: minimal state display
Item {
    id: taskPanel
    property var verseApp: null

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
        color: "#081018"
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
                font.pixelSize: 13
                wrapMode: Text.WordWrap
                width: parent.width
            }

            Text {
                id: resultText
                text: taskPanel._result
                color: taskPanel._resultColor
                font.pixelSize: 12
                wrapMode: Text.WordWrap
                width: parent.width
                visible: taskPanel._result.length > 0
            }
        }
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