#version 440
// V1 aurora veil fragment shader (Qt 6 RHI pipeline, compiled with qsb).
// Visual port of the approved HTML reference's curtain field: four
// uneven organic curtain formations with soft internal rays/filaments,
// natural overlap, lower merging braid, and shared atmospheric haze.
// Stars and ambient glow stay in their existing Verse layers.
//
// Coordinates are derived from qt_TexCoord0 (top-left origin, backend
// agnostic) and re-expressed in the reference's bottom-up,
// aspect-normalized space, so geometry matches the HTML prototype on any
// graphics API and at any device pixel ratio.
layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;
layout(std140, binding = 0) uniform buf {
    mat4 qt_Matrix;
    float qt_Opacity;
    float uAspect;
    float uTime;
    float uIntensity;
    float uEnergy;
};

float h21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vn(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1.0, 0.0)), f.x),
               mix(h21(i + vec2(0.0, 1.0)), h21(i + vec2(1.0, 1.0)), f.x), f.y);
}

float fbm(vec2 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        s += a * vn(p);
        p = p * 2.03 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return s;
}

vec3 auroraCol(float y) {
    vec3 top = vec3(0.12, 1.0, 0.58);
    vec3 teal = vec3(0.05, 0.85, 0.78);
    vec3 cyan = vec3(0.10, 0.62, 0.98);
    vec3 pur = vec3(0.55, 0.20, 0.85);
    vec3 c = mix(pur, cyan, smoothstep(0.10, 0.42, y));
    c = mix(c, teal, smoothstep(0.45, 0.65, y));
    c = mix(c, top, smoothstep(0.66, 0.86, y));
    return c;
}

float curtainField(vec2 p, float y, float t, float baseX,
                   float targetX, float seed, float width, float topReach) {
    float merge = smoothstep(0.04, 0.58, y);
    float lower = 1.0 - merge;
    float largeBend = fbm(vec2(p.x * 0.70 + seed * 1.73 + t * 0.018,
                               y * 0.62 - seed * 2.1 - t * 0.012));
    float mediumBend = fbm(vec2(p.x * 1.45 - seed * 2.4 - t * 0.028,
                                y * 1.15 + seed * 1.9 + t * 0.017));
    float path = baseX
        + (0.30 + 0.10 * largeBend)
            * sin(y * (1.35 + seed * 0.07) + seed * 5.1 + t * 0.09)
        + (0.13 + 0.05 * mediumBend)
            * sin(y * (3.1 + seed * 0.11) - t * 0.13 + seed * 2.4);
    float organicNoise = fbm(vec2(seed * 3.1 + y * 1.8, t * 0.022 + seed));
    path += (organicNoise - 0.5) * 0.16;
    float mergeWave = 0.045 * sin(y * 6.0 + t * 0.12 + seed * 8.0)
        + 0.035 * sin(y * 11.0 - t * 0.08 + seed * 4.0);
    float center = mix(targetX + mergeWave, path, merge);
    float widen = width * (1.0 + 0.95 * lower);
    float vertical = smoothstep(0.02, 0.15, y);
    vertical *= 1.0 - smoothstep(topReach - 0.13, topReach, y);
    float bodyNoise = fbm(vec2(center * 2.0 + seed, y * 1.35 + t * 0.015));
    float body = 0.62 + 0.38 * bodyNoise;
    float acc = 0.0;
    float dx = p.x - center;
    float broad = exp(-(dx * dx) / (2.0 * widen * widen));
    acc += broad * 0.72 * body;
    for (int j = 0; j < 7; j++) {
        float fj = float(j);
        float r1 = h21(vec2(fj + seed * 3.7, seed + 2.1));
        float r2 = h21(vec2(fj + seed * 5.9, seed + 8.7));
        float r3 = h21(vec2(fj + seed * 8.2, seed + 15.4));
        float slot = (fj / 6.0 - 0.5) * 1.45;
        float rayOffset = slot * widen
            + (0.09 + 0.05 * r1)
            * sin(y * (2.2 + 2.5 * r2) + t * (0.07 + 0.10 * r3) + seed * 7.0);
        float rayCenter = center + rayOffset;
        float rayWidth = widen * mix(0.045, 0.11, r2);
        float rd = p.x - rayCenter;
        float ray = exp(-(rd * rd) / (rayWidth * rayWidth * 2.0));
        float pulse = 0.66 + 0.34 * sin(y * (5.0 + 4.0 * r3)
            - t * (0.08 + 0.12 * r1) + r2 * 20.0);
        float broken = 0.55 + 0.45 * fbm(vec2(fj * 1.7 + seed, y * 2.7 - t * 0.028));
        acc += ray * pulse * broken * mix(0.40, 1.0, r3);
    }
    float fine = fbm(vec2(p.x * 11.0 + seed * 4.0 + t * 0.025,
                          y * 2.2 - t * 0.02));
    float filament = exp(-abs(dx) * 18.0);
    acc += filament * smoothstep(0.32, 0.78, fine) * 0.22;
    float joining = lower * (0.65 + 0.35 * fbm(vec2(p.x * 2.7 + seed, y * 5.0 - t * 0.018)));
    acc += broad * joining * 0.34;
    return acc * vertical;
}

void main() {
    vec2 uv = vec2(qt_TexCoord0.x, 1.0 - qt_TexCoord0.y);
    vec2 p = vec2((uv.x - 0.5) * uAspect, uv.y - 0.5);
    float uvy = uv.y;
    float t = uTime;
    float w1 = fbm(vec2(p.x * 0.85 + t * 0.030, p.y * 0.65 - t * 0.020));
    float w2 = fbm(vec2(p.x * 1.7 - t * 0.040, p.y * 1.35 + 3.0 + t * 0.030));
    float c1 = curtainField(p, uvy, t, -0.82, -0.26, 1.7, 0.14, 0.91);
    float c2 = curtainField(p, uvy, t, -0.29, -0.09, 4.1, 0.17, 0.83);
    float c3 = curtainField(p, uvy, t, 0.26, 0.10, 7.3, 0.15, 0.94);
    float c4 = curtainField(p, uvy, t, 0.78, 0.29, 10.6, 0.13, 0.86);
    float curtains = c1 + c2 + c3 + c4;
    float lower = 1.0 - smoothstep(0.10, 0.42, uvy);
    float bridgeNoise = fbm(vec2(p.x * 2.0 + t * 0.012, uvy * 5.4 - t * 0.018));
    float bridgeShape = smoothstep(0.16, 0.78, bridgeNoise);
    float bridge = lower * bridgeShape;
    bridge *= smoothstep(0.96, 0.08, abs(p.x));
    float haze = smoothstep(0.28, 0.82, fbm(vec2(p.x * 1.7 + w1, uvy * 0.95 - t * 0.018)));
    float env = smoothstep(0.0, 0.18, uvy) * (1.0 - smoothstep(0.92, 1.04, uvy));
    float intensity = curtains * 0.54 + bridge * 0.26 + haze * 0.11;
    intensity *= env;
    float braid = lower * lower * bridgeShape * 0.22;
    intensity += braid;
    float ycol = clamp(uvy + (w1 - 0.5) * 0.24 + (w2 - 0.5) * 0.10, 0.0, 1.0);
    vec3 col = auroraCol(ycol) * intensity;
    col *= uIntensity * (0.70 + 0.50 * uEnergy) * 1.15;
    col = vec3(1.0) - exp(-col * 1.5);
    float alpha = clamp(intensity * uIntensity, 0.0, 1.0);
    fragColor = vec4(col * alpha * qt_Opacity, alpha);
}
