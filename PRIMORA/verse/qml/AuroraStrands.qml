import QtQuick

// M3 fine strands: thin drifting arcs above the main ribbons, same
// curtain language at a smaller scale. Slow phase drift keeps them alive
// without drawing attention away from the ribbons and the presence.
Item {
    id: strands
    objectName: "strands"

    property real energy: 0.62
    property double t0: 0
    property var lines: []

    function lcg(seed) {
        var s = seed >>> 0
        return function() {
            s = (s * 1664525 + 1013904223) % 4294967296
            return s / 4294967296
        }
    }

    function regenerate() {
        var rnd = lcg(777)
        var colors = ["#3fe0c5", "#4a8fe0", "#8c7cf5", "#9fd8ff"]
        var arr = []
        for (var i = 0; i < 14; i++) {
            arr.push({
                y: 0.08 + rnd() * 0.55,
                x0: rnd() * 0.9,
                len: 0.10 + rnd() * 0.22,
                color: colors[i % colors.length],
                phase: rnd() * 6.2832,
                speed: 0.05 + rnd() * 0.08,
                alpha: 0.08 + rnd() * 0.10
            })
        }
        lines = arr
    }

    function elapsedSeconds() {
        return (Date.now() - t0) / 1000.0
    }

    function repaint() {
        strandCanvas.requestPaint()
    }

    Component.onCompleted: {
        t0 = Date.now()
        regenerate()
    }
    onWidthChanged: strandCanvas.requestPaint()
    onHeightChanged: strandCanvas.requestPaint()

    Canvas {
        id: strandCanvas
        anchors.fill: parent
        renderTarget: Canvas.FramebufferObject

        onPaint: {
            var ctx = getContext("2d")
            var w = width
            var h = height
            var t = strands.elapsedSeconds()
            var energyBoost = 0.7 + 0.6 * strands.energy
            ctx.clearRect(0, 0, w, h)
            var list = strands.lines
            for (var i = 0; i < list.length; i++) {
                var ln = list[i]
                var drift = Math.sin(t * ln.speed + ln.phase) * 0.03
                var x0 = (ln.x0 + drift) * w
                var x1 = x0 + ln.len * w
                var y = ln.y * h
                var a = ln.alpha * energyBoost
                if (a > 0.5) a = 0.5
                ctx.strokeStyle = ln.color + Math.round(a * 255).toString(16).padStart(2, "0")
                ctx.lineWidth = 1.2
                ctx.beginPath()
                ctx.moveTo(x0, y)
                ctx.quadraticCurveTo(
                    (x0 + x1) / 2, y - 14 - 10 * Math.sin(t * ln.speed * 1.3 + ln.phase),
                    x1, y + 6 * Math.sin(t * ln.speed + ln.phase * 2.0)
                )
                ctx.stroke()
            }
        }
    }
}
