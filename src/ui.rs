use crate::tool::Shape;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
const MAX_PUSH_FORCE_N: f32 = 15_000.0;
const MAX_SIMULATION_TIME_SCALE: f32 = 31_557_600.0;
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
    pub menu_open: bool,
    pub settings_open: bool,
    /// Simulation seconds per wall-clock second. 1.0 is real time.
    pub simulation_time_scale: f32,
    pub walk_speed_mps: f32,
    pub look_sensitivity: f32,
    paused_before_menu: bool,
}

pub enum UiAction {
    Shape(Shape),
    GenerateWorld(String),
    Resume,
    Settings,
    Back,
    Exit,
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
            menu_open: false,
            settings_open: false,
            simulation_time_scale: 1.0,
            walk_speed_mps: 4.5,
            look_sensitivity: 0.005,
            paused_before_menu: false,
        }
    }
    pub fn direction(&self) -> Vec3 {
        self.direction
    }
    pub fn start(&self) -> Vec3 {
        self.start
    }
    pub fn click(&mut self, x: f32, y: f32) -> Option<UiAction> {
        if self.world_active && !self.menu_open {
            return None;
        }
        if self.menu_open {
            if self.settings_open {
                if (470.0..=810.0).contains(&x) && (570.0..=626.0).contains(&y) {
                    self.settings_open = false;
                    return Some(UiAction::Back);
                }
                if (480.0..=800.0).contains(&x) {
                    for (top, center) in [(298.0, 310.0), (378.0, 390.0), (458.0, 470.0)] {
                        if (top..=top + 38.0).contains(&y) {
                            self.pointer(x, center);
                            return None;
                        }
                    }
                }
                return None;
            }
            if (470.0..=810.0).contains(&x) {
                if (318.0..=370.0).contains(&y) {
                    self.close_menu();
                    return Some(UiAction::Resume);
                }
                if (382.0..=434.0).contains(&y) {
                    self.settings_open = true;
                    return Some(UiAction::Settings);
                }
                if (446.0..=498.0).contains(&y) {
                    return Some(UiAction::Exit);
                }
            }
            return None;
        }
        if (20.0..=280.0).contains(&x) && (646.0..=666.0).contains(&y) {
            self.seed_editing = true;
            return None;
        }
        if (20.0..=280.0).contains(&x) && (670.0..=694.0).contains(&y) {
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
    pub fn toggle_pause_menu(&mut self) {
        if self.menu_open {
            if self.settings_open {
                self.settings_open = false;
            } else {
                self.close_menu();
            }
        } else {
            self.seed_editing = false;
            self.paused_before_menu = self.paused;
            self.paused = true;
            self.menu_open = true;
        }
    }
    pub fn close_menu(&mut self) {
        self.menu_open = false;
        self.settings_open = false;
        self.paused = self.paused_before_menu;
    }
    pub fn pointer(&mut self, x: f32, y: f32) -> Option<Shape> {
        if self.menu_open {
            if self.settings_open && (480.0..=800.0).contains(&x) {
                let t = ((x - 490.0) / 300.0).clamp(0.0, 1.0);
                if (298.0..=336.0).contains(&y) {
                    self.simulation_time_scale = MAX_SIMULATION_TIME_SCALE.powf(t);
                } else if (378.0..=416.0).contains(&y) {
                    self.walk_speed_mps = 1.0 + 9.0 * t;
                } else if (458.0..=496.0).contains(&y) {
                    self.look_sensitivity = 0.001 + 0.011 * t;
                }
            }
            return None;
        }
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
    pub fn draw(
        &self,
        broken: usize,
        clusters: usize,
        fps: f32,
        sim_ms: f32,
        world_age_years: Option<f32>,
    ) -> Vec<Rect> {
        let mut r = Vec::new();
        if !self.world_active {
        rect(
            &mut r,
            [10.0, 10.0, 285.0, 710.0],
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
        if self.world_active {
            text(&mut r, 24., 530., "ESC MENU  P PAUSE  R RESET", 1.1);
            text(&mut r, 24., 548., "WASD WALK  SPACE JUMP  SHIFT RUN", 0.85);
            text(&mut r, 24., 566., "MOUSE LOOK  F FREE CAMERA", 0.9);
        } else {
            text(&mut r, 24., 530., "SPACE PAUSE   R RESET", 1.2);
            text(&mut r, 24., 548., "WASD FLY  SPACE/CTRL UP/DOWN  RMB LOOK", 0.8);
        }
        text(&mut r, 24., 590., "P START / STOP RECORDING", 1.);
        text(&mut r, 24., 608., "REPLAY LOOPS AT 1X", 1.);
        text(&mut r, 24., 626., "MAP 10 KM2 | REV 1 CM | LOAD 1 KM", 1.1);
        rect(
            &mut r,
            [20., 646., 260., 20.],
            if self.seed_editing {
                [0.13, 0.22, 0.25, 1.]
            } else {
                [0.09, 0.14, 0.18, 1.]
            },
        );
        let seed_display = self.seed_text.to_ascii_uppercase();
        text(&mut r, 28., 652., &seed_display, 1.3);
        if self.seed_editing {
            let cursor_x = (28. + seed_display.len() as f32 * 9.0).min(270.0);
            rect(&mut r, [cursor_x, 650., 2., 15.], [0.3, 0.85, 0.65, 1.]);
        }
        rect(
            &mut r,
            [20., 670., 260., 24.],
            if self.world_active {
                [0.12, 0.38, 0.33, 1.]
            } else {
                [0.14, 0.23, 0.29, 1.]
            },
        );
        text(
            &mut r,
            50.,
            676.,
            if self.world_active { "REGENERATE WORLD" } else { "GENERATE WORLD" },
            1.4,
        );
        if let Some(age) = world_age_years {
            text(
                &mut r,
                24.,
                699.,
                &format!("AGE {age:.0} Y | SIM {:.2}X", self.simulation_time_scale),
                1.0,
            );
        }
        }
        if self.menu_open {
            rect(&mut r, [0., 0., 1280., 720.], [0.015, 0.025, 0.04, 0.78]);
            rect(&mut r, [430., 125., 420., 520.], [0.055, 0.085, 0.105, 0.99]);
            rect(&mut r, [430., 145., 420., 4.], [0.22, 0.68, 0.57, 1.]);
            if self.settings_open {
                text(&mut r, 482., 184., "WORLD SETTINGS", 2.0);
                text(&mut r, 490., 262., &format!("SIM SPEED {:.2}X", self.simulation_time_scale), 1.15);
                settings_slider(
                    &mut r,
                    (self.simulation_time_scale.ln() / MAX_SIMULATION_TIME_SCALE.ln())
                        .clamp(0.0, 1.0),
                    310.,
                );
                text(&mut r, 490., 324., "1X = REAL TIME", 0.9);
                text(&mut r, 653., 324., "MAX = 1 YEAR / SEC", 0.9);
                text(&mut r, 490., 342., "WALK SPEED M/S", 1.15);
                text(&mut r, 700., 342., &format!("{:.1}", self.walk_speed_mps), 1.15);
                settings_slider(&mut r, (self.walk_speed_mps - 1.0) / 9.0, 390.);
                text(&mut r, 490., 422., "LOOK SENSITIVITY", 1.15);
                text(&mut r, 700., 422., &format!("{:.3}", self.look_sensitivity), 1.15);
                settings_slider(&mut r, (self.look_sensitivity - 0.001) / 0.011, 470.);
                menu_button(&mut r, 470., 570., 340., 52., "BACK", false);
            } else {
                text(&mut r, 482., 202., "PAUSED", 2.4);
                text(&mut r, 482., 246., "WORLD STATE IS PRESERVED", 1.1);
                menu_button(&mut r, 470., 318., 340., 52., "RESUME", true);
                menu_button(&mut r, 470., 382., 340., 52., "SETTINGS", false);
                menu_button(&mut r, 470., 446., 340., 52., "EXIT", false);
                text(&mut r, 482., 530., "ESC TO RESUME", 1.0);
                text(&mut r, 482., 552., "CTRL CROUCH", 1.0);
                text(&mut r, 482., 574., "F6 20M PHYSICS EXPERIMENT", 1.0);
            }
        }
        r
    }
}
fn menu_button(out: &mut Vec<Rect>, x: f32, y: f32, w: f32, h: f32, label: &str, primary: bool) {
    rect(
        out,
        [x, y, w, h],
        if primary { [0.15, 0.52, 0.44, 1.] } else { [0.12, 0.19, 0.23, 1.] },
    );
    text(out, x + 24., y + 16., label, 1.5);
}
fn settings_slider(out: &mut Vec<Rect>, fraction: f32, y: f32) {
    let t = fraction.clamp(0.0, 1.0);
    rect(out, [490., y, 300., 8.], [0.18, 0.26, 0.29, 1.]);
    rect(out, [490., y, 300. * t, 8.], [0.18, 0.65, 0.56, 1.]);
    rect(out, [486. + 300. * t, y - 5., 8., 18.], [0.5, 0.9, 0.75, 1.]);
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
