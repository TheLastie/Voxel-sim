struct Camera { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> camera: Camera;
struct Input {
    @location(0) vertex: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) center: vec3<f32>,
    @location(3) scale: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) rotation: vec4<f32>,
    @location(6) surface: f32,
};
struct Output {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) @interpolate(flat) surface: f32,
};
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
@vertex fn vs(i: Input) -> Output {
    var o: Output;
    let world_position = i.center + rotate(i.rotation, i.vertex * i.scale);
    o.position = camera.view_proj * vec4<f32>(world_position, 1.0);
    o.normal = rotate(i.rotation, i.normal);
    o.world_position = world_position;
    o.surface = i.surface;
    o.color = i.color;
    return o;
}
fn hash(p: vec2<f32>) -> f32 {
    var q = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    q += vec3<f32>(dot(q, q.yzx + vec3<f32>(33.33)));
    return fract((q.x + q.y) * q.z);
}
fn noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = fract(p);
    let u = f*f*(vec2<f32>(3.0)-2.0*f);
    return mix(mix(hash(cell),hash(cell+vec2<f32>(1.0,0.0)),u.x),
        mix(hash(cell+vec2<f32>(0.0,1.0)),hash(cell+vec2<f32>(1.0,1.0)),u.x),u.y);
}
@fragment fn fs(i: Output) -> @location(0) vec4<f32> {
    let p = i.world_position;
    // Suppress subpixel detail rather than introducing distant shimmer.
    let footprint = max(length(dpdx(p)),length(dpdy(p)));
    let fine_visibility = 1.0-smoothstep(0.008,0.07,footprint);
    let medium_visibility = 1.0-smoothstep(0.08,0.7,footprint);
    var color = i.color.rgb;
    if i.surface > 1.5 {
        let grain = noise(vec2<f32>((p.x+p.y)*38.0,p.z*1.6));
        let fissure = smoothstep(0.60,0.82,grain);
        let mottling = noise(vec2<f32>((p.x-p.y)*5.0,p.z*9.0));
        color *= 1.0 + medium_visibility*(mottling-0.5)*0.30;
        color *= 1.0-fine_visibility*fissure*0.48;
        color += vec3<f32>(0.04,0.025,0.012)*fine_visibility*(1.0-grain);
    } else if i.surface > 0.5 {
        let patches = noise(p.xy*0.32 + vec2<f32>(17.0,43.0));
        let soil = smoothstep(0.48,0.78,patches)*0.65;
        color = mix(color,vec3<f32>(0.28,0.20,0.12),soil);
        let clumps = noise(p.xy*5.0);
        let grains = noise(p.xy*85.0);
        color *= 1.0 + (clumps-0.5)*0.32*medium_visibility
            + (grains-0.5)*0.30*fine_visibility;
        let pebbles = smoothstep(0.80,0.94,grains)*fine_visibility*0.38;
        color = mix(color,vec3<f32>(0.43,0.41,0.34),pebbles);
    }
    let light = 0.35 + 0.65 * max(dot(normalize(i.normal),
        normalize(vec3<f32>(-0.4,-0.6,1.0))),0.0);
    return vec4<f32>(color*light,i.color.a);
}
