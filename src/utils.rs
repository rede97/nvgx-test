use std::ops::Mul;

use image::{ImageBuffer, Rgb};
use ndarray::ArrayView3;
use nvgx::Rect;

#[allow(unused)]
pub fn save_ndarray_as_png(array: ArrayView3<f32>, path: &str) -> Result<(), image::ImageError> {
    let (_, height, width) = array.dim();
    let mut img = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (array[[0, y as usize, x as usize]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        let g = (array[[1, y as usize, x as usize]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        let b = (array[[2, y as usize, x as usize]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        *pixel = Rgb([r, g, b]);
    }

    img.save(path)
}

#[allow(unused)]
pub fn save_ndarray_as_png_t(array: ArrayView3<f32>, path: &str) -> Result<(), image::ImageError> {
    let (height, width, _) = array.dim();
    let mut img = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (array[[y as usize, x as usize, 0]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        let g = (array[[y as usize, x as usize, 1]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        let b = (array[[y as usize, x as usize, 2]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        *pixel = Rgb([r, g, b]);
    }

    img.save(path)
}

pub fn scale_rect(r: Rect, scale: (f32, f32)) -> Rect {
    return Rect {
        xy: (r.xy.x * scale.0, r.xy.y * scale.1).into(),
        size: (r.size.width * scale.0, r.size.height * scale.1).into(),
    };
}

pub fn sigmoid(x: &f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}