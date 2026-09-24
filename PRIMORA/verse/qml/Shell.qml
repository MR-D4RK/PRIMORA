import QtQuick
import QtQuick.Window

// Verse shell: one persistent Universe plus state-driven overlays. The
// window is created hidden and shown exactly once from Python (after mode
// flags are applied) so startup never flashes an unstyled frame. The
// awakening experience is the dormant Universe itself becoming alive:
// Universe.dormancy eases down while Aurora warms up, and the existing
// warmup result drives the final settle to AWAKE (or DEGRADED).
Window {
    id: root
    property string verseState: "starting"
    property string verseError: ""
    property string verseMode: "portal"
    width: 960
    height: 640
    visible: false
    title: "PRIMORA Verse"
    color: "#05070d"

    // Escape key works even when child items have focus
    Shortcut {
        sequence: StandardKey.Quit
        onActivated: {
            verseApp.shutdown()
        }
    }
    Shortcut {
        sequence: "Escape"
        onActivated: {
            verseApp.shutdown()
        }
    }

    // Alt+F4 / window close button -> same shutdown path
    onClosing: {
        if (verseApp) {
            verseApp.shutdown()
            close.accepted = false // we handle close ourselves
        }
    }

    // The Universe is always present: dormant while warming, alive when
    // awake. No scene swap, so waking is a continuous emergence.
    Universe {
        id: universe
        anchors.fill: parent
        dormancy: 1.0
    }

    // Slow emergence while warming. This never completes on its own: it
    // drifts toward a calm almost-awake rest state and waits there for the
    // real warmup result. No progress, no percentages, no spinners.
    NumberAnimation {
        id: dormantDrift
        target: universe
        property: "dormancy"
        from: 1.0
        to: 0.30
        duration: 150000
        easing.type: Easing.InOutSine
    }

    // Final settle once the real warmup reports success.
    NumberAnimation {
        id: wakeSettle
        target: universe
        property: "dormancy"
        to: 0.0
        duration: 3000
        easing.type: Easing.InOutSine
    }

    onVerseStateChanged: {
        if (verseState === "awake") {
            dormantDrift.stop()
            wakeSettle.start()
        } else if (verseState === "awakening") {
            wakeSettle.stop()
            dormantDrift.start()
        } else {
            dormantDrift.stop()
            wakeSettle.stop()
        }
    }

    Component.onCompleted: {
        if (verseState !== "awake") {
            dormantDrift.start()
        }
    }

    // Failure readout only (DEGRADED preserves its error display).
    Awakening {
        anchors.fill: parent
        errorText: root.verseError
    }

    // Overlay UI (conversation + task panel) - only visible when awake
    Item {
        id: overlay
        anchors.fill: parent
        visible: root.verseState === "awake"

        // Conversation area - bottom portion
        Conversation {
            id: conversationView
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: parent.height * 0.35
        }

        // Task panel - top right
        TaskPanel {
            id: taskPanelView
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 16
            width: 280
            height: 120
        }
    }
}

