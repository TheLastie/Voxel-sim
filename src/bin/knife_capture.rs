fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/knife_capture.csv".into());
    let speed = std::env::args()
        .nth(2)
        .map(|v| v.parse().expect("speed must be numeric"))
        .unwrap_or(6.0);
    let frames = std::env::args()
        .nth(3)
        .map(|v| v.parse().expect("frame count must be numeric"))
        .unwrap_or(100);
    let stride = std::env::args()
        .nth(4)
        .map(|v| v.parse().expect("capture stride must be numeric"))
        .unwrap_or(1);
    let angle = std::env::args()
        .nth(5)
        .map(|v| v.parse().expect("attack angle must be numeric"))
        .unwrap_or(0.0);
    voxel_physics_lab::experiments::capture_knife(&path, stride, speed, frames, angle)
        .expect("capture knife solver states");
    println!("Captured knife solver states to {path} at {speed} m/s for {frames} frames (stride {stride}, attack angle {angle}°)");
}
