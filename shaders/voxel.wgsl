struct Camera { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> camera: Camera;
struct Input {
    @location(0) vertex: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) center: vec3<f32>,
    @location(3) scale: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) rotation: vec4<f32>,
};
struct Output { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> };
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
@vertex fn vs(i: Input) -> Output {
    var o: Output;
    o.position = camera.view_proj * vec4<f32>(i.center + rotate(i.rotation, i.vertex * i.scale), 1.0);
    let n = rotate(i.rotation, i.normal);
    let light = 0.35 + 0.65 * max(dot(n, normalize(vec3<f32>(-0.4, -0.6, 1.0))), 0.0);
    o.color = vec4<f32>(i.color.rgb * light, i.color.a);
    return o;
}
@fragment fn fs(i: Output) -> @location(0) vec4<f32> { return i.color; }
