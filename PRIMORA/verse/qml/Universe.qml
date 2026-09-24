import QtQuick

// M3 Universe root: depth-ordered layers sharing one energy value and two
// paint timers. Background -> far stars -> ribbons -> strands -> soft
// illumination -> central presence. A slow camera drift moves the whole
// world a few pixels so the space feels alive without demanding attention.
// `active` is the M4 sleep/wake hook: when false, timers stop and the
// presence breathing pauses, leaving the last frame resting in place.
// `dormancy` is the awakening hook: 1.0 is deep sleep (almost-black space,
// faint stars, no aurora), 0.0 is fully awake. Shell eases it down while
// Aurora warms up and settles it to 0 on the real warmup result.
Item {
    id: universe
    objectName: "universeRoot"

    property bool active: true
    property real dormancy: 0.0
    property real wakefulness: 1.0 - dormancy
    property real energy: 0.62
    property double t0: 0

    Component.onCompleted: {
        t0 = Date.now()
    }

    function elapsedSeconds() {
        return (Date.now() - t0) / 1000.0
    }

    Rectangle {
        anchors.fill: parent
        color: "#020409"
    }

    Item {
        id: camera
        anchors.centerIn: parent
        width: parent.width + 48
        height: parent.height + 48

        // Subtle orbital drift: ~20px over ~50s per axis, out of phase.
        NumberAnimation on x {
            running: universe.active
            loops: Animation.Infinite
            from: -10; to: 10; duration: 52000
            easing.type: Easing.InOutSine
        }
        NumberAnimation on y {
            running: universe.active
            loops: Animation.Infinite
            from: 8; to: -8; duration: 67000
            easing.type: Easing.InOutSine
        }

        Starfield {
            anchors.fill: parent
            energy: universe.energy
            dim: 0.12 + 0.88 * universe.wakefulness
        }

        AuroraRibbons {
            anchors.fill: parent
            energy: universe.energy
            opacity: 0.05 + 0.95 * universe.wakefulness
        }

        AuroraStrands {
            anchors.fill: parent
            energy: universe.energy
            opacity: 0.04 + 0.96 * universe.wakefulness
        }

        // Soft illumination the presence casts onto nearby space.
        Canvas {
            id: auraLight
            objectName: "auraLight"
            anchors.fill: parent
            renderTarget: Canvas.FramebufferObject
            opacity: 0.08 + 0.92 * universe.wakefulness

            function repaint() {
                requestPaint()
            }

            onPaint: {
                var ctx = getContext("2d")
                var cx = width / 2
                var cy = height / 2
                var r = Math.min(width, height) * 0.42
                var a = 0.05 + 0.08 * universe.energy
                ctx.clearRect(0, 0, width, height)
                var g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r)
                g.addColorStop(0, "rgba(110,170,240," + a.toFixed(3) + ")")
                g.addColorStop(1, "rgba(110,170,240,0)")
                ctx.fillStyle = g
                ctx.fillRect(0, 0, width, height)
            }
        }

        Presence {
            objectName: "presenceItem"
            anchors.centerIn: parent
            width: Math.min(parent.width, parent.height) * 0.30
            height: width
            energy: universe.energy
            active: universe.active
            opacity: 0.22 + 0.78 * universe.wakefulness
        }
    }

    // Fast tick: evolving layers + shared energy (~30fps).
    Timer {
        id: fastTimer
        objectName: "fastTimer"
        interval: 33
        repeat: true
        running: universe.active
        triggeredOnStart: true
        onTriggered: {
            var t = universe.elapsedSeconds()
            var w = universe.wakefulness
            universe.energy = 0.05 + w * (0.57
                + 0.10 * Math.sin(t * 0.11)
                + 0.05 * Math.sin(t * 0.043 + 1.7))
            var kids = camera.children
            for (var i = 0; i < kids.length; i++) {
                if (kids[i] !== auraLight && typeof kids[i].repaint === "function") {
                    kids[i].repaint()
                }
            }
            auraLight.repaint()
        }
    }

    // Slow tick: star twinkle (~7fps is plenty for distant stars).
    Timer {
        id: slowTimer
        objectName: "slowTimer"
        interval: 140
        repeat: true
        running: universe.active
        onTriggered: {
            var kids = camera.children
            for (var i = 0; i < kids.length; i++) {
                if (kids[i].objectName === "starfield"
                        && typeof kids[i].repaint === "function") {
                    kids[i].repaint()
                }
            }
        }
    }
}
