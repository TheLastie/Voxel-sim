use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Rect {
    pub rect: [f32; 4],
    pub color: [f32; 4],
}

pub fn draw_text(out: &mut Vec<Rect>, x: f32, y: f32, text: &str, scale: f32) {
    #[cfg(windows)]
    native_font::draw(out, x, y, text, scale);
    #[cfg(not(windows))]
    bitmap_text(out, x, y, text, scale);
}

#[cfg(not(windows))]
fn rect(out: &mut Vec<Rect>, rect: [f32; 4], color: [f32; 4]) {
    out.push(Rect { rect, color });
}
#[cfg(not(windows))]
fn bitmap_text(out: &mut Vec<Rect>, x: f32, y: f32, s: &str, scale: f32) {
    let scale = scale.max(1.0);
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

#[cfg(windows)]
mod native_font {
    use super::{glyph, Rect};
    use std::{
        collections::HashMap,
        ffi::c_void,
        ptr::null_mut,
        sync::{Mutex, OnceLock},
    };

    type Handle = *mut c_void;

    #[repr(C)]
    struct BitmapInfoHeader {
        size: u32,
        width: i32,
        height: i32,
        planes: u16,
        bit_count: u16,
        compression: u32,
        size_image: u32,
        x_pels_per_meter: i32,
        y_pels_per_meter: i32,
        colors_used: u32,
        colors_important: u32,
    }

    #[repr(C)]
    struct BitmapInfo {
        header: BitmapInfoHeader,
        colors: [u32; 1],
    }

    #[repr(C)]
    struct TextSize {
        width: i32,
        height: i32,
    }

    #[derive(Clone)]
    struct Run {
        x: i32,
        y: i32,
        width: i32,
        coverage: u8,
    }

    #[derive(Clone)]
    struct Glyph {
        advance: f32,
        runs: Vec<Run>,
    }

    static CACHE: OnceLock<Mutex<HashMap<(char, i32), Glyph>>> = OnceLock::new();

    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn CreateCompatibleDC(dc: Handle) -> Handle;
        fn DeleteDC(dc: Handle) -> i32;
        fn CreateFontW(
            height: i32,
            width: i32,
            escapement: i32,
            orientation: i32,
            weight: i32,
            italic: u32,
            underline: u32,
            strike_out: u32,
            charset: u32,
            output_precision: u32,
            clip_precision: u32,
            quality: u32,
            pitch_and_family: u32,
            face: *const u16,
        ) -> Handle;
        fn SelectObject(dc: Handle, object: Handle) -> Handle;
        fn DeleteObject(object: Handle) -> i32;
        fn SetBkMode(dc: Handle, mode: i32) -> i32;
        fn SetTextColor(dc: Handle, color: u32) -> u32;
        fn GdiFlush() -> i32;
        fn TextOutW(dc: Handle, x: i32, y: i32, text: *const u16, count: i32) -> i32;
        fn GetTextExtentPoint32W(
            dc: Handle,
            text: *const u16,
            count: i32,
            size: *mut TextSize,
        ) -> i32;
        fn CreateDIBSection(
            dc: Handle,
            info: *const BitmapInfo,
            usage: u32,
            bits: *mut *mut c_void,
            section: Handle,
            offset: u32,
        ) -> Handle;
    }

    pub fn draw(out: &mut Vec<Rect>, mut x: f32, y: f32, text: &str, scale: f32) {
        let height = (scale * 14.0).round().clamp(12.0, 26.0) as i32;
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().expect("font cache poisoned");
        for character in text.chars() {
            let cached = cache.entry((character, height)).or_insert_with(|| {
                rasterize(character, height).unwrap_or_else(|| fallback(character))
            });
            for run in &cached.runs {
                out.push(Rect {
                    rect: [x + run.x as f32, y + run.y as f32, run.width as f32, 1.0],
                    color: [0.83, 0.91, 0.93, run.coverage as f32 / 255.0],
                });
            }
            x += cached.advance;
        }
    }

    fn rasterize(character: char, height: i32) -> Option<Glyph> {
        const SUPERSAMPLE: i32 = 4;
        const WIDTH: i32 = 144;
        const EXTRA_HEIGHT: i32 = 4;
        let raster_height = (height + EXTRA_HEIGHT) * SUPERSAMPLE;
        let dc = unsafe { CreateCompatibleDC(null_mut()) };
        if dc.is_null() {
            return None;
        }
        let info = BitmapInfo {
            header: BitmapInfoHeader {
                size: std::mem::size_of::<BitmapInfoHeader>() as u32,
                width: WIDTH,
                height: -raster_height,
                planes: 1,
                bit_count: 32,
                compression: 0,
                size_image: 0,
                x_pels_per_meter: 0,
                y_pels_per_meter: 0,
                colors_used: 0,
                colors_important: 0,
            },
            colors: [0],
        };
        let mut pixels = null_mut();
        let bitmap = unsafe { CreateDIBSection(dc, &info, 0, &mut pixels, null_mut(), 0) };
        let face: Vec<u16> = "Segoe UI".encode_utf16().chain([0]).collect();
        let font = unsafe {
            CreateFontW(
                -height * SUPERSAMPLE,
                0,
                0,
                0,
                400,
                0,
                0,
                0,
                1,
                4,
                0,
                4,
                0,
                face.as_ptr(),
            )
        };
        if bitmap.is_null() || pixels.is_null() || font.is_null() {
            unsafe {
                if !bitmap.is_null() {
                    DeleteObject(bitmap);
                }
                if !font.is_null() {
                    DeleteObject(font);
                }
                DeleteDC(dc);
            }
            return None;
        }
        let old_bitmap = unsafe { SelectObject(dc, bitmap) };
        let old_font = unsafe { SelectObject(dc, font) };
        unsafe {
            std::ptr::write_bytes(pixels.cast::<u8>(), 0, (WIDTH * raster_height * 4) as usize);
            SetBkMode(dc, 1);
            SetTextColor(dc, 0x00ff_ffff);
        }
        let encoded: Vec<u16> = character.encode_utf16(&mut [0; 2]).to_vec();
        let mut size = TextSize {
            width: height * SUPERSAMPLE / 2,
            height: height * SUPERSAMPLE,
        };
        let _ =
            unsafe { GetTextExtentPoint32W(dc, encoded.as_ptr(), encoded.len() as i32, &mut size) };
        let _ = unsafe {
            TextOutW(
                dc,
                SUPERSAMPLE,
                SUPERSAMPLE,
                encoded.as_ptr(),
                encoded.len() as i32,
            )
        };
        unsafe {
            GdiFlush();
        }

        let width = WIDTH as usize;
        let raster_rows = raster_height as usize;
        let bytes =
            unsafe { std::slice::from_raw_parts(pixels.cast::<u8>(), width * raster_rows * 4) };
        let rows = (height + EXTRA_HEIGHT) as usize;
        let logical_width = width / SUPERSAMPLE as usize;
        let coverage_at = |column: usize, row: usize| {
            let mut total = 0u32;
            for sy in 0..SUPERSAMPLE as usize {
                for sx in 0..SUPERSAMPLE as usize {
                    let offset = (((row * SUPERSAMPLE as usize + sy) * width)
                        + column * SUPERSAMPLE as usize
                        + sx)
                        * 4;
                    total += bytes[offset..offset + 3].iter().copied().max().unwrap_or(0) as u32;
                }
            }
            (total / (SUPERSAMPLE * SUPERSAMPLE) as u32) as u8
        };
        let mut runs = Vec::new();
        for row in 0..rows {
            let mut col = 0usize;
            while col < logical_width {
                let coverage = coverage_at(col, row);
                if coverage < 8 {
                    col += 1;
                    continue;
                }
                let start = col;
                col += 1;
                while col < logical_width && coverage_at(col, row) == coverage {
                    col += 1;
                }
                runs.push(Run {
                    x: start as i32 - 1,
                    y: row as i32 - 1,
                    width: (col - start) as i32,
                    coverage,
                });
            }
        }
        unsafe {
            SelectObject(dc, old_font);
            SelectObject(dc, old_bitmap);
            DeleteObject(font);
            DeleteObject(bitmap);
            DeleteDC(dc);
        }
        Some(Glyph {
            advance: (size.width.max(2) as f32 / SUPERSAMPLE as f32),
            runs,
        })
    }

    fn fallback(character: char) -> Glyph {
        let rows = glyph(character);
        let mut runs = Vec::new();
        for (y, row) in rows.iter().copied().enumerate() {
            for x in 0..5 {
                if row & (1 << (4 - x)) != 0 {
                    runs.push(Run {
                        x,
                        y: y as i32,
                        width: 1,
                        coverage: 255,
                    });
                }
            }
        }
        Glyph { advance: 6.0, runs }
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
