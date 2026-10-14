import QtQuick

// M3 Universe root: depth-ordered layers sharing one energy value and two
// paint timers. Background -> far stars -> aurora veil -> soft atmospheric
// haze. A slow camera drift moves the whole world a few pixels so the space
// feels alive without demanding attention.
// `active` is the M4 sleep/wake hook: when false, timers stop, leaving the
// last frame resting in place.
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

    // Staggered emergence: each layer fades in over its own wakefulness
    // window (smoothstep), so awakening reads as phases rather than a
    // uniform brightening. Diffuse haze next, ray curtains later.
    // At dormancy 1 all veil layers are exactly 0.
    function emerge(lo, hi) {
        var t = (wakefulness - lo) / (hi - lo)
        t = Math.min(1, Math.max(0, t))
        return t * t * (3 - 2 * t)
    }
    // Star reveal clock: wall-clock seconds since this Universe appeared.
    // Driven by the fast timer below. Deliberately independent of dormancy
    // and model readiness: stars own their timeline (first light ~2 s,
    // full field ~12 s), so they never wait on model loading.
    property real starReveal: 0.0
    property real glowDim: emerge(0.10, 0.55)
    property real ribbonDim: emerge(0.30, 0.75)

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
            dim: 1.0
            reveal: universe.starReveal
        }

        AuroraRibbons {
            anchors.fill: parent
            energy: universe.energy
            intensity: universe.ribbonDim
        }

        // Soft atmospheric haze over the lower sky.
        Canvas {
            id: auraLight
            objectName: "auraLight"
            anchors.fill: parent
            renderTarget: Canvas.FramebufferObject
            opacity: universe.glowDim

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
            universe.energy = 0.03 + w * (0.59
                + 0.10 * Math.sin(t * 0.11)
                + 0.05 * Math.sin(t * 0.043 + 1.7))
            // Star reveal timeline: first light ~2 s after Verse appears,
            // full population ~12 s, eased. Wall-clock only — never gated
            // on dormancy or model readiness.
            var r = (t - 2.0) / 10.0
            r = Math.min(1, Math.max(0, r))
            universe.starReveal = r * r * (3 - 2 * r)
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
