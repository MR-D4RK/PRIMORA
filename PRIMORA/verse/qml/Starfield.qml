import QtQuick

// M3 starfield: many tiny stars plus a few brighter ones with soft glow.
// Positions come from a fixed-seed generator so every launch shares the
// same deterministic layout; twinkle phases evolve with elapsed time.
// Brightness follows the shared Universe energy, so stars visibly react
// when Aurora's presence strengthens or softens.
Item {
    id: starfield
    objectName: "starfield"

    property int starCount: 220
    property int brightCount: 8
    property real energy: 0.62
    property real dim: 1.0
    // Reveal progress 0..1, driven by the Universe star clock (wall-clock
    // startup time, never model progress). Default full for standalone use.
    property real reveal: 1.0
    property double t0: 0
    property var stars: []

    // Per-star reveal threshold in [0, 1), derived from the star's own
    // seeded twinkle phase: deterministic, spread across the population,
    // independent of position, size, or brightness.
    function revealStart(phase) {
        var s = (phase * 0.15915494) % 1.0
        return s < 0 ? s + 1.0 : s
    }

    // Visibility gate 0..1 for one star at the current reveal progress.
    // Smooth 0.15-wide fade per star; 0 before its threshold, 1 at full
    // reveal for every phase. No timers, no model state involved.
    function gateFor(phase) {
        var start = revealStart(phase)
        var end = start + 0.15
        if (end > 1.0) {
            end = 1.0
        }
        var t = (reveal - start) / (end - start)
        t = Math.min(1, Math.max(0, t))
        return t * t * (3 - 2 * t)
    }

    function lcg(seed) {
        var s = seed >>> 0
        return function() {
            s = (s * 1664525 + 1013904223) % 4294967296
            return s / 4294967296
        }
    }

    function regenerate() {
        var rnd = lcg(1337)
        var arr = []
        var i
        for (i = 0; i < starCount; i++) {
            arr.push({
                x: rnd(), y: rnd(),
                r: 0.4 + rnd() * 0.9,
                phase: rnd() * 6.2832,
                speed: 0.4 + rnd() * 1.4,
                base: 0.35 + rnd() * 0.4,
                bright: false
            })
        }
        for (i = 0; i < brightCount; i++) {
            arr.push({
                x: rnd(), y: rnd(),
                r: 1.6 + rnd() * 1.2,
                phase: rnd() * 6.2832,
                speed: 0.25 + rnd() * 0.5,
                base: 0.8 + rnd() * 0.2,
                bright: true
            })
        }
        stars = arr
    }

    function elapsedSeconds() {
        return (Date.now() - t0) / 1000.0
    }

    function repaint() {
        starCanvas.requestPaint()
    }

    Component.onCompleted: {
        t0 = Date.now()
        regenerate()
    }
    onWidthChanged: starCanvas.requestPaint()
    onHeightChanged: starCanvas.requestPaint()

    Canvas {
        id: starCanvas
        anchors.fill: parent

        onPaint: {
            var ctx = getContext("2d")
            var w = width
            var h = height
            var t = starfield.elapsedSeconds()
            var glow = (0.45 + 0.9 * starfield.energy) * starfield.dim
            ctx.clearRect(0, 0, w, h)
            var list = starfield.stars
            for (var i = 0; i < list.length; i++) {
                var s = list[i]
                var tw = 0.55 + 0.45 * Math.sin(t * s.speed + s.phase)
                var a = s.base * tw * glow * gateFor(s.phase)
                if (a > 1.0) a = 1.0
                var x = s.x * w
                var y = s.y * h
                if (s.bright) {
                    var g = ctx.createRadialGradient(x, y, 0, x, y, s.r * 5.0)
                    g.addColorStop(0, "rgba(220,235,255," + (0.35 * a).toFixed(3) + ")")
                    g.addColorStop(1, "rgba(220,235,255,0)")
                    ctx.fillStyle = g
                    ctx.beginPath()
                    ctx.arc(x, y, s.r * 5.0, 0, 6.2832)
                    ctx.fill()
                }
                ctx.fillStyle = "rgba(234,243,246," + a.toFixed(3) + ")"
                ctx.beginPath()
                ctx.arc(x, y, s.r, 0, 6.2832)
                ctx.fill()
            }
        }
    }
}
