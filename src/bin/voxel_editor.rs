pub use voxel_physics_lab::{bond, scene, tool};

#[path = "../main.rs"]
mod app;

pub use app::ui;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::run_editor()
}
