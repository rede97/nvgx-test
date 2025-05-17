#[macro_use]
#[allow(unused)]
extern crate anyhow;

mod demo;
mod yolov5_face;

use anyhow::Error;
use fast_image_resize::{PixelType, images::ImageRef};
use num_traits::AsPrimitive;
use nvgx::*;
use yolov5_face::YoloV5Face;

fn padding_fit_img<N1: AsPrimitive<f32>, N2: AsPrimitive<f32>>(
    img_size: (N1, N1),
    display_size: (N2, N2),
) -> (f32, f32) {
    let img_size: (f32, f32) = (img_size.0.as_(), img_size.1.as_());
    let display_size: (f32, f32) = (display_size.0.as_(), display_size.1.as_());
    let fit_width = display_size.1 * img_size.0 / img_size.1;
    if fit_width <= display_size.0 {
        return (fit_width, display_size.1);
    } else {
        return (display_size.0, display_size.0 * img_size.1 / img_size.0);
    }
}

struct DemoDraw {
    img_size: Option<(ImageId, (u32, u32))>,
    camera: kamera::Camera,
    yolov5n_face: YoloV5Face,
}

impl<R: RendererDevice> demo::Demo<R> for DemoDraw {
    fn init(&mut self, ctx: &mut Context<R>, _scale_factor: f32) -> Result<(), Error> {
        ctx.create_font_from_file("roboto", demo::FONT_PATH)?;
        self.camera.start();
        Ok(())
    }

    fn update(&mut self, width: f32, height: f32, ctx: &mut Context<R>) -> anyhow::Result<()> {
        let Some(frame) = self.camera.wait_for_frame() else {
            return Ok(());
        };

        let cap_size = frame.size_u32();
        let img_size = if let Some((img, img_size)) = self.img_size {
            let img = if img_size != cap_size {
                ctx.delete_image(img)?;
                ctx.create_image(
                    cap_size.0,
                    cap_size.1,
                    TextureType::BGRA,
                    ImageFlags::REPEATX | ImageFlags::REPEATY,
                    Some(frame.data().data_u8()),
                )?
            } else {
                ctx.update_image(img, frame.data().data_u8(), None)?;
                img
            };
            (img, cap_size)
        } else {
            let img = ctx.create_image(
                cap_size.0,
                cap_size.1,
                TextureType::BGRA,
                ImageFlags::REPEATX | ImageFlags::REPEATY,
                Some(frame.data().data_u8()),
            )?;
            (img, cap_size)
        };

        let buffer = frame.data();
        let src_img =
            ImageRef::new(cap_size.0, cap_size.1, buffer.data_u8(), PixelType::U8x3).unwrap();
        self.yolov5n_face.proc(&src_img).unwrap();

        self.img_size = Some(img_size);
        let fill_size = padding_fit_img(img_size.1, (width, height));
        let xy = ((width - fill_size.0) / 2.0, (height - fill_size.1) / 2.0);
        let square_width = f32::min(fill_size.0, fill_size.1);
        let square_xy = ((width - square_width) / 2.0, (height - square_width) / 2.0);

        ctx.begin_path();
        ctx.fill_paint({
            ImagePattern {
                img: img_size.0,
                center: xy.into(),
                size: fill_size.into(),
                angle: 0.0,
                alpha: 1.0,
            }
        });
        ctx.rect(Rect {
            xy: xy.into(),
            size: fill_size.into(),
        });
        ctx.fill()?;

        ctx.begin_path();
        ctx.rect(Rect {
            xy: square_xy.into(),
            size: (square_width, square_width).into(),
        });
        ctx.stroke_paint(nvgx::Color::rgb(1.0, 0.4, 0.3));
        ctx.stroke()?;

        Ok(())
    }
}

fn main() {
    demo::run(
        DemoDraw {
            img_size: None,
            camera: kamera::Camera::new_default_device(),
            yolov5n_face: YoloV5Face::new("weights/yolov5n-face-relu.onnx").unwrap(),
        },
        "demo-draw",
    );
}
