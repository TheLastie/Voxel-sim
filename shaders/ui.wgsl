struct Screen { size: vec2<f32>, padding: vec2<f32> };
@group(0) @binding(0) var<uniform> screen: Screen;
struct Output { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32, @location(0) rect: vec4<f32>, @location(1) color: vec4<f32>) -> Output {
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.,0.), vec2<f32>(1.,0.), vec2<f32>(0.,1.), vec2<f32>(0.,1.), vec2<f32>(1.,0.), vec2<f32>(1.,1.));
    let p = rect.xy + corners[i] * rect.zw;
    var o: Output;
    o.position = vec4<f32>(p.x / screen.size.x * 2. - 1., 1. - p.y / screen.size.y * 2., 0., 1.);
    o.color = color;
    return o;
}
@fragment fn fs(i: Output) -> @location(0) vec4<f32> { return i.color; }
