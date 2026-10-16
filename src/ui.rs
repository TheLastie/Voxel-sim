use crate::tool::Shape;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
const MAX_PUSH_FORCE_N: f32 = 15_000.0;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Rect {
    pub rect: [f32; 4],
    pub color: [f32; 4],
}
pub struct Ui {
    pub speed: f32,
    pub start: Vec3,
    pub direction: Vec3,
    pub force_newtons: f32,
    pub paused: bool,
    pub show_bonds: bool,
    pub selected: Shape,
    pub seed_text: String,
    pub seed_editing: bool,
    pub world_active: bool,
}

pub enum UiAction {
    Shape(Shape),
    GenerateWorld(String),
}

impl Ui {
    pub fn new(speed: f32, start: Vec3, direction: Vec3) -> Self {
        Self {
            speed,
            start,
            direction: direction.normalize(),
            force_newtons: 0.0,
            paused: false,
            show_bonds: false,
            selected: Shape::Wedge,
            seed_text: "1001".to_owned(),
            seed_editing: false,
            world_active: false,
        }
    }
    pub fn direction(&self) -> Vec3 {
        self.direction
    }
    pub fn start(&self) -> Vec3 {
        self.start
    }
    pub fn click(&mut self, x: f32, y: f32) -> Option<UiAction> {
        if (20.0..=280.0).contains(&x) && (615.0..=644.0).contains(&y) {
            self.seed_editing = true;
            return None;
        }
        if (20.0..=280.0).contains(&x) && (650.0..=678.0).contains(&y) {
            self.seed_editing = false;
            return Some(UiAction::GenerateWorld(self.seed_text.clone()));
        }
        self.seed_editing = false;
        self.pointer(x, y).map(UiAction::Shape)
    }
    pub fn append_seed_text(&mut self, text: &str) {
        if !self.seed_editing {
            return;
        }
        for c in text.chars() {
            if self.seed_text.chars().count() >= 24 {
                break;
            }
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                self.seed_text.push(c);
            }
        }
    }
    pub fn erase_seed_character(&mut self) {
        if self.seed_editing {
            self.seed_text.pop();
        }
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
        if (246.0..278.0).contains(&y) {
            let t = ((x - 25.0) / 245.0).clamp(0.0, 1.0);
            self.force_newtons = t * t * MAX_PUSH_FORCE_N;
        }
        None
    }
    pub fn draw(&self, broken: usize, clusters: usize, fps: f32, sim_ms: f32) -> Vec<Rect> {
        let mut r = Vec::new();
        rect(
            &mut r,
            [10.0, 10.0, 285.0, 680.0],
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
        let force_fraction = (self.force_newtons / MAX_PUSH_FORCE_N)
            .sqrt()
            .clamp(0.0, 1.0);
        text(&mut r, 24., 218., "CONSTANT PUSH FORCE", 1.5);
        text(&mut r, 24., 237., &format!("{:.0} N", self.force_newtons), 1.6);
        rect(&mut r, [25., 260., 245., 10.], [0.2, 0.26, 0.3, 1.]);
        rect(
            &mut r,
            [25., 260., 245. * force_fraction, 10.],
            [0.18, 0.65, 0.56, 1.],
        );
        rect(
            &mut r,
            [25. + force_fraction * 245. - 4., 256., 8., 18.],
            [0.3, 0.85, 0.65, 1.],
        );
        text(&mut r, 25., 280., "0 N", 1.1);
        text(&mut r, 216., 280., "15000 N", 1.1);
        text(&mut r, 24., 300., "0 N CLEARS FORCE; TOOL COASTS", 0.9);
        let (axis, sign) = if self.direction.x.abs() >= self.direction.y.abs() {
            ("X", if self.direction.x < 0.0 { "-" } else { "+" })
        } else {
            ("Y", if self.direction.y < 0.0 { "-" } else { "+" })
        };
        text(&mut r, 24., 322., &format!("PUSH AXIS {sign}{axis}"), 1.2);
        text(
            &mut r,
            24.,
            470.,
            &format!("BROKEN {} CLUSTERS {}", broken, clusters),
            1.2,
        );
        text(
            &mut r,
            24.,
            490.,
            &format!("FPS {:.1} STEP {:.1} MS", fps, sim_ms),
            1.2,
        );
        text(&mut r, 24., 530., "SPACE PAUSE   R RESET", 1.2);
        text(&mut r, 24., 548., "B BONDS  RMB ORBIT  WHEEL ZOOM", 1.);
        text(&mut r, 24., 572., "P START / STOP RECORDING", 1.);
        text(&mut r, 24., 590., "REPLAY LOOPS AT 1X", 1.);
        text(&mut r, 24., 612., "WORLD SEED (100 M2)", 1.3);
        rect(
            &mut r,
            [20., 632., 260., 24.],
            if self.seed_editing {
                [0.13, 0.22, 0.25, 1.]
            } else {
                [0.09, 0.14, 0.18, 1.]
            },
        );
        let seed_display = self.seed_text.to_ascii_uppercase();
        text(&mut r, 28., 640., &seed_display, 1.5);
        if self.seed_editing {
            let cursor_x = (28. + seed_display.len() as f32 * 9.0).min(270.0);
            rect(&mut r, [cursor_x, 638., 2., 15.], [0.3, 0.85, 0.65, 1.]);
        }
        rect(
            &mut r,
            [20., 660., 260., 26.],
            if self.world_active {
                [0.12, 0.38, 0.33, 1.]
            } else {
                [0.14, 0.23, 0.29, 1.]
            },
        );
        text(
            &mut r,
            50.,
            667.,
            if self.world_active { "REGENERATE WORLD" } else { "GENERATE WORLD" },
            1.4,
        );
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
