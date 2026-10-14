fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/guided-knife-cut.csv".into());
    let frames = std::env::args()
        .nth(2)
        .map(|value| value.parse().expect("frame count must be an integer"))
        .unwrap_or(17_600);
    let stride = std::env::args()
        .nth(3)
        .map(|value| value.parse().expect("capture stride must be an integer"))
        .unwrap_or(100);
    let force = std::env::args()
        .nth(4)
        .map(|value| value.parse().expect("force must be numeric"))
        .unwrap_or(200.0);

    voxel_physics_lab::experiments::capture_guided_knife(&path, stride, frames, force)
        .expect("capture guided knife solver states");
    println!(
        "Captured guided knife states to {path}: {frames} steps, stride {stride}, force {force} N"
    );
}
