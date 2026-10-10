use glam::Vec3;
pub const EMPTY: u32 = u32::MAX;
#[derive(Clone, Copy, Debug)]
pub struct Bond {
    pub neighbor: u32,
    pub plastic_strain: [f32; 3],
    pub damage: f32,
    pub equivalent_plastic_strain: f32,
    pub plastic_work: f32,
    pub fatigue_damage: f32,
    pub previous_stress_ratio: f32,
    pub reversal_stress_ratio: f32,
    pub loading_increment: f32,
}
impl Default for Bond {
    fn default() -> Self {
        Self {
            neighbor: EMPTY,
            plastic_strain: [0.0; 3],
            damage: 0.0,
            equivalent_plastic_strain: 0.0,
            plastic_work: 0.0,
            fatigue_damage: 0.0,
            previous_stress_ratio: 0.0,
            reversal_stress_ratio: 0.0,
            loading_increment: 0.0,
        }
    }
}
/// Solver workspace, not persistent material state. Reset lambda per substep.
#[derive(Clone, Copy, Debug, Default)]
pub struct BondWorkspace {
    pub lambda: Vec3,
    pub rest_neighbor: u32,
    pub last_force: Vec3,
}
