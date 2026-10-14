import QtQuick

// V1 aurora veil: four organic curtain formations with soft internal
// rays/filaments, natural overlap and lower merging. Rendered by the
// precompiled AuroraVeil.frag.qsb shader package (Qt 6 RHI pipeline;
// inline GLSL strings are not compiled on any Qt 6 backend). Visual port
// of the approved HTML reference's curtain field; stars and ambient glow
// stay in their existing layers.
Item {
    id: ribbons
    objectName: "ribbons"

    property real energy: 0.62
    property real intensity: 1.0
    property double t0: 0
    property real uT: 0.0

    function elapsedSeconds() {
        return (Date.now() - t0) / 1000.0
    }

    // Called by the Universe fast timer (~30fps); advances shader time.
    // Keeps the existing paint-driver contract: curtains evolve whenever
    // the scene is active, even at zero intensity (no pop-in on reveal).
    function repaint() {
        uT = elapsedSeconds()
    }

    Component.onCompleted: {
        t0 = Date.now()
        uT = 0.0
    }

    ShaderEffect {
        anchors.fill: parent
        fragmentShader: "AuroraVeil.frag.qsb"
        property real uAspect: width / Math.max(1.0, height)
        property real uTime: ribbons.uT
        property real uIntensity: ribbons.intensity
        property real uEnergy: ribbons.energy
    }
}
