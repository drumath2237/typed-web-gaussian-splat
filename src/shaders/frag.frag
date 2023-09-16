precision mediump float;

varying vec4 vColor;
varying vec3 vConic;
varying vec2 vCenter;
uniform vec2 viewport;
uniform vec2 focal;

void main() {
    vec2 d = (vCenter - 2.0 * (gl_FragCoord.xy / viewport - vec2(0.5, 0.5))) * viewport * 0.5;
    float power = -0.5 * (vConic.x * d.x * d.x + vConic.z * d.y * d.y) - vConic.y * d.x * d.y;
    if(power > 0.0)
        discard;
    float alpha = min(0.99, vColor.a * exp(power));
    if(alpha < 0.02)
        discard;

    gl_FragColor = vec4(alpha * vColor.rgb, alpha);
}
