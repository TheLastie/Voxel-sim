use crate::tool::Shape;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Rect {
    pub rect: [f32; 4],
    pub color: [f32; 4],
}
pub struct Ui {
    pub speed: f32,
    pub azimuth: f32,
    pub elevation: f32,
    pub height: f32,
    pub lateral: f32,
    pub distance: f32,
    pub longitudinal: f32,
    pub paused: bool,
    pub show_bonds: bool,
    pub selected: Shape,
}
impl Ui {
    pub fn new(speed: f32, start: Vec3) -> Self {
        Self {
            speed,
            azimuth: 0.0,
            elevation: 0.0,
            height: start.z,
            lateral: start.y,
            distance: -start.x,
            longitudinal: 0.0,
            paused: false,
            show_bonds: false,
            selected: Shape::Wedge,
        }
    }
    pub fn direction(&self) -> Vec3 {
        Vec3::new(
            self.elevation.cos() * self.azimuth.cos(),
            self.elevation.cos() * self.azimuth.sin(),
            self.elevation.sin(),
        )
    }
    pub fn start(&self) -> Vec3 {
        Vec3::new(self.longitudinal, self.lateral, self.height) - self.direction() * self.distance
    }
    pub fn pointer(&mut self, x: f32, y: f32) -> Option<Shape> {
        if !(20.0..=280.0).contains(&x) {
            return None;
        }
        for (n, shape) in [Shape::Wedge, Shape::Blade, Shape::Rod, Shape::Rounded]
            .into_iter()
            .enumerate()
        {
            let top = 50.0 + n as f32 * 32.0;
            if (top..top + 26.0).contains(&y) {
                self.selected = shape;
                return Some(shape);
            }
        }
        for index in 0..7 {
            let top = 205.0 + index as f32 * 42.0;
            if (top..top + 16.0).contains(&y) {
                let t = ((x - 25.0) / 245.0).clamp(0.0, 1.0);
                match index {
                    0 => self.speed = 0.5 + t * 19.5,
                    1 => self.azimuth = (t - 0.5) * std::f32::consts::TAU,
                    2 => self.elevation = (t - 0.5) * std::f32::consts::PI,
                    3 => self.height = 0.03 + t * 3.2,
                    4 => self.lateral = (t - 0.5) * 0.6,
                    5 => self.distance = 0.45 + t * 0.8,
                    _ => self.longitudinal = (t - 0.5) * 1.0,
                }
            }
        }
        None
    }
    pub fn draw(&self, broken: usize, clusters: usize, fps: f32, sim_ms: f32) -> Vec<Rect> {
        let mut r = Vec::new();
        rect(
            &mut r,
            [10.0, 10.0, 285.0, 590.0],
            [0.025, 0.04, 0.06, 0.94],
        );
        text(&mut r, 24., 24., "VOXEL MECHANICS LAB", 1.5);
        for (i, name) in [
            "1 УДАРИТЬ ТОПОРОМ",
            "2 УДАРИТЬ НОЖОМ",
            "3 УДАРИТЬ ПАЛКОЙ",
            "4 УДАРИТЬ КАМНЕМ",
        ]
        .iter()
        .enumerate()
        {
            let y = 50. + i as f32 * 32.;
            rect(&mut r, [20., y, 260., 26.], [0.14, 0.23, 0.29, 1.]);
            text(&mut r, 30., y + 7., name, 1.5);
        }
        let labels = [
            format!("SPEED {:.1} M/S", self.speed),
            format!("AZIMUTH {:.0} DEG", self.azimuth.to_degrees()),
            format!("ELEVATION {:.0} DEG", self.elevation.to_degrees()),
            format!("HEIGHT {:.2} M", self.height),
            format!("LATERAL {:.2} M", self.lateral),
            format!("START DIST {:.2} M", self.distance),
            format!("CONTACT X {:.2} M", self.longitudinal),
        ];
        let values = [
            (self.speed - 0.5) / 19.5,
            self.azimuth / std::f32::consts::TAU + 0.5,
            self.elevation / std::f32::consts::PI + 0.5,
            (self.height - 0.03) / 3.2,
            self.lateral / 0.6 + 0.5,
            (self.distance - 0.45) / 0.8,
            self.longitudinal + 0.5,
        ];
        for i in 0..7 {
            let y = 205. + i as f32 * 42.;
            text(&mut r, 24., y - 15., &labels[i], 1.25);
            rect(&mut r, [25., y, 245., 8.], [0.2, 0.26, 0.3, 1.]);
            rect(
                &mut r,
                [25. + values[i].clamp(0., 1.) * 245. - 3., y - 3., 6., 14.],
                [0.3, 0.85, 0.65, 1.],
            );
        }
        text(
            &mut r,
            24.,
            510.,
            &format!("BROKEN {} CLUSTERS {}", broken, clusters),
            1.2,
        );
        text(
            &mut r,
            24.,
            528.,
            &format!("FPS {:.1} STEP {:.1} MS", fps, sim_ms),
            1.2,
        );
        text(&mut r, 24., 549., "SPACE PAUSE   R RESET", 1.2);
        text(&mut r, 24., 566., "B BONDS  RMB ORBIT  WHEEL ZOOM", 1.);
        r
    }
}
fn rect(out: &mut Vec<Rect>, rect: [f32; 4], color: [f32; 4]) {
    out.push(Rect { rect, color });
}
fn text(out: &mut Vec<Rect>, x: f32, y: f32, s: &str, scale: f32) {
    for (index, c) in s.chars().enumerate() {
        let glyph = glyph(c);
        for row in 0..7 {
            for col in 0..5 {
                if glyph[row] & (1 << (4 - col)) != 0 {
                    rect(
                        out,
                        [
                            x + (index * 6 + col) as f32 * scale,
                            y + row as f32 * scale,
                            scale,
                            scale,
                        ],
                        [0.83, 0.91, 0.93, 1.],
                    );
                }
            }
        }
    }
}
fn glyph(c: char) -> [u8; 7] {
    match c {
        'А' | 'A' => [14, 17, 17, 31, 17, 17, 17],
        'В' | 'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'Е' | 'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'Н' | 'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'J' => [7, 2, 2, 2, 18, 18, 12],
        'К' | 'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'М' | 'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 21, 19, 17, 17, 17],
        'О' | 'O' => [14, 17, 17, 17, 17, 17, 14],
        'Р' | 'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'Т' | 'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        'У' => [17, 17, 17, 15, 1, 17, 14],
        'Д' => [6, 10, 10, 10, 10, 31, 17],
        'И' => [17, 17, 19, 21, 25, 17, 17],
        'Й' => [10, 4, 17, 19, 21, 25, 17],
        'Ь' => [16, 16, 16, 30, 17, 17, 30],
        'П' => [31, 17, 17, 17, 17, 17, 17],
        'Ж' => [21, 21, 21, 14, 21, 21, 21],
        'Л' => [7, 9, 9, 9, 9, 9, 17],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        '.' => [0, 0, 0, 0, 0, 6, 6],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '/' => [1, 2, 2, 4, 8, 8, 16],
        _ => [0; 7],
    }
}
