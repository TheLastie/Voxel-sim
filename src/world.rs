use glam::Vec3;

pub const WORLD_SIDE_M: f32 = 10.0;
pub const WORLD_CELL_M: f32 = 0.1;
pub const WORLD_SIDE_CELLS: usize = (WORLD_SIDE_M / WORLD_CELL_M) as usize;

#[derive(Clone, Copy, Debug)]
pub struct TerrainVoxel {
    pub center: Vec3,
    pub size: Vec3,
    pub color: [f32; 4],
}

#[derive(Debug)]
pub struct GeneratedWorld {
    pub voxels: Vec<TerrainVoxel>,
}

impl GeneratedWorld {
    /// Build a deterministic 10 m × 10 m terrain heightfield from the seed.
    /// These coarse visual voxels are not attached to the centimeter physics lattice.
    pub fn from_seed(seed: &str) -> Self {
        let seed_hash = hash_seed(seed);
        let mut voxels = Vec::with_capacity(WORLD_SIDE_CELLS * WORLD_SIDE_CELLS);
        for y in 0..WORLD_SIDE_CELLS {
            for x in 0..WORLD_SIDE_CELLS {
                let large = value_noise(seed_hash, x as f32 / 34.0, y as f32 / 34.0);
                let medium = value_noise(seed_hash ^ 0xA076_1D64_78BD_642F, x as f32 / 13.0, y as f32 / 13.0);
                let fine = value_noise(seed_hash ^ 0xE703_7ED1_A0B4_28DB, x as f32 / 5.0, y as f32 / 5.0);
                let height = (0.55 + large * 0.72 + medium * 0.26 + fine * 0.08).max(0.16);
                let variation = (large * 0.5 + medium * 0.35 + fine * 0.15).clamp(-1.0, 1.0);
                let color = [
                    0.31 + variation * 0.055,
                    0.34 + variation * 0.08,
                    0.19 + variation * 0.045,
                    1.0,
                ];
                voxels.push(TerrainVoxel {
                    center: Vec3::new(
                        (x as f32 + 0.5) * WORLD_CELL_M - WORLD_SIDE_M * 0.5,
                        (y as f32 + 0.5) * WORLD_CELL_M - WORLD_SIDE_M * 0.5,
                        height * 0.5,
                    ),
                    size: Vec3::new(WORLD_CELL_M, WORLD_CELL_M, height),
                    color,
                });
            }
        }
        Self { voxels }
    }
}

fn hash_seed(seed: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in seed.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

fn lattice_hash(seed: u64, x: i32, y: i32) -> u64 {
    let mut value = seed
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn value_noise(seed: u64, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let tx = smooth(x - x.floor());
    let ty = smooth(y - y.floor());
    let a = unit_noise(lattice_hash(seed, x0, y0));
    let b = unit_noise(lattice_hash(seed, x0 + 1, y0));
    let c = unit_noise(lattice_hash(seed, x0, y0 + 1));
    let d = unit_noise(lattice_hash(seed, x0 + 1, y0 + 1));
    lerp(lerp(a, b, tx), lerp(c, d, tx), ty)
}

fn unit_noise(value: u64) -> f32 {
    (value >> 40) as f32 / 8_388_607.5 - 1.0
}

fn smooth(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
