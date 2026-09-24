import QtQuick

// M3 aurora ribbons: large flowing bands with a real ray-curtain look.
// Each band is built from many thin vertical gradient streaks whose control
// points drift with slow phase offsets, plus one brighter core pass.
// Palette: cyan / blue / violet. All motion is a pure function of elapsed
// time over fixed-seed parameters, so the sky evolves continuously yet
// deterministically.
Item {
    id: ribbons
    objectName: "ribbons"

    property real energy: 0.62
    property double t0: 0
    property var bands: []

    function lcg(seed) {
        var s = seed >>> 0
        return function() {
            s = (s * 1664525 + 1013904223) % 4294967296
            return s / 4294967296
        }
    }

    function regenerate() {
        var rnd = lcg(4242)
        var colors = ["#3fe0c5", "#4a8fe0", "#8c7cf5"]
        var arr = []
        for (var b = 0; b < 3; b++) {
            arr.push({
                cx: 0.22 + b * 0.28 + (rnd() - 0.5) * 0.06,
                width: 0.16 + rnd() * 0.10,
                color: colors[b],
                phase: rnd() * 6.2832,
                drift: 0.008 + rnd() * 0.010,
                sway: 0.015 + rnd() * 0.020,
                streaks: 26,
                alpha: 0.055 + rnd() * 0.035
            })
        }
        bands = arr
    }

    function elapsedSeconds() {
        return (Date.now() - t0) / 1000.0
    }

    function repaint() {
        ribbonCanvas.requestPaint()
    }

    Component.onCompleted: {
        t0 = Date.now()
        regenerate()
    }
    onWidthChanged: ribbonCanvas.requestPaint()
    onHeightChanged: ribbonCanvas.requestPaint()

    Canvas {
        id: ribbonCanvas
        anchors.fill: parent
        renderTarget: Canvas.FramebufferObject

        onPaint: {
            var ctx = getContext("2d")
            var w = width
            var h = height
            var t = ribbons.elapsedSeconds()
            var energyBoost = 0.7 + 0.6 * ribbons.energy
            ctx.clearRect(0, 0, w, h)
            var list = ribbons.bands
            for (var b = 0; b < list.length; b++) {
                var band = list[b]
                var cx = (band.cx + band.drift * Math.sin(t * 0.05 + band.phase)) * w
                var half = band.width * w * 0.5
                var s
                for (s = 0; s < band.streaks; s++) {
                    var f = (s + 0.5) / band.streaks - 0.5
                    var edge = 1.0 - Math.abs(f) * 2.0
                    var x = cx + f * half * 2.0
                    var sway = Math.sin(t * 0.11 + band.phase + s * 0.35) * band.sway * w
                    var a = band.alpha * edge * edge * energyBoost
                    if (a <= 0.003) continue
                    var grad = ctx.createLinearGradient(0, 0, 0, h)
                    grad.addColorStop(0, "rgba(0,0,0,0)")
                    grad.addColorStop(0.35, band.color + Math.round(a * 255).toString(16).padStart(2, "0"))
                    grad.addColorStop(0.8, band.color + Math.round(a * 1.6 > 1 ? 255 : a * 1.6 * 255).toString(16).padStart(2, "0"))
                    grad.addColorStop(1, "rgba(0,0,0,0)")
                    ctx.strokeStyle = grad
                    ctx.lineWidth = 5.0 + edge * 7.0
                    ctx.beginPath()
                    ctx.moveTo(x, -10)
                    ctx.quadraticCurveTo(
                        x + sway, h * 0.35,
                        x + sway * 0.4, h * 0.7
                    )
                    ctx.quadraticCurveTo(x, h * 0.9, x - sway * 0.3, h + 10)
                    ctx.stroke()
                }
                var coreA = 0.16 * energyBoost
                if (coreA > 1.0) coreA = 1.0
                var core = ctx.createLinearGradient(0, 0, 0, h)
                core.addColorStop(0, "rgba(0,0,0,0)")
                core.addColorStop(0.4, band.color + Math.round(coreA * 255).toString(16).padStart(2, "0"))
                core.addColorStop(1, "rgba(0,0,0,0)")
                ctx.strokeStyle = core
                ctx.lineWidth = 2.0
                ctx.beginPath()
                ctx.moveTo(cx, -10)
                ctx.quadraticCurveTo(
                    cx + Math.sin(t * 0.11 + band.phase) * band.sway * w,
                    h * 0.5, cx, h + 10
                )
                ctx.stroke()
            }
        }
    }
}
