import QtQuick

// M3 central Aurora presence: an abstract, intentional glow — layered
// radial light, gentle breathing, no character or avatar forms.
// (The detailed eye/iris design is explicitly a later milestone.)
Item {
    id: presence
    objectName: "presence"

    property real energy: 0.62
    property bool active: true

    function repaint() {
        glowCanvas.requestPaint()
    }

    Canvas {
        id: glowCanvas
        anchors.fill: parent
        renderTarget: Canvas.FramebufferObject

        onPaint: {
            var ctx = getContext("2d")
            var w = width
            var h = height
            var cx = w / 2
            var cy = h / 2
            var base = Math.min(w, h)
            var e = presence.energy
            ctx.clearRect(0, 0, w, h)

            var halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, base * 0.30)
            halo.addColorStop(0, "rgba(140,124,245," + (0.16 * e + 0.06).toFixed(3) + ")")
            halo.addColorStop(1, "rgba(140,124,245,0)")
            ctx.fillStyle = halo
            ctx.fillRect(0, 0, w, h)

            var mid = ctx.createRadialGradient(cx, cy, 0, cx, cy, base * 0.14)
            mid.addColorStop(0, "rgba(120,220,255," + (0.35 * e + 0.10).toFixed(3) + ")")
            mid.addColorStop(1, "rgba(120,220,255,0)")
            ctx.fillStyle = mid
            ctx.fillRect(0, 0, w, h)

            var core = ctx.createRadialGradient(cx, cy, 0, cx, cy, base * 0.05)
            core.addColorStop(0, "rgba(240,250,255," + (0.55 * e + 0.25).toFixed(3) + ")")
            core.addColorStop(1, "rgba(240,250,255,0)")
            ctx.fillStyle = core
            ctx.fillRect(0, 0, w, h)
        }
    }

    // Slow breathing scale; paused when the Universe is not active so the
    // presence rests instead of animating (M4 sleep/wake builds on this).
    SequentialAnimation on scale {
        running: presence.active
        loops: Animation.Infinite
        NumberAnimation { from: 0.97; to: 1.03; duration: 2750; easing.type: Easing.InOutSine }
        NumberAnimation { from: 1.03; to: 0.97; duration: 2750; easing.type: Easing.InOutSine }
    }
}
