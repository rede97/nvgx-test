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
        let frame_data = frame.data();

        let cap_size = frame.size_u32();

        let img_update = match self.img_size {
            Some((img, img_size)) if img_size == cap_size => {
                ctx.update_image(img, frame_data.data_u8(), None)?;
                Some(img)
            }
            Some((img, _)) => {
                ctx.delete_image(img)?;
                None
            }
            _ => None,
        };
        let img = match img_update {
            Some(img) => img,
            _ => {
                let img = ctx.create_image(
                    cap_size.0,
                    cap_size.1,
                    TextureType::BGRA,
                    ImageFlags::REPEATX | ImageFlags::REPEATY,
                    Some(frame_data.data_u8()),
                )?;
                img
            }
        };
        self.img_size = Some((img, cap_size));

        let fill_size = padding_fit_img(cap_size, (width, height));
        let xy = ((width - fill_size.0) / 2.0, (height - fill_size.1) / 2.0);
        let square_width = f32::min(fill_size.0, fill_size.1);
        let square_xy = ((width - square_width) / 2.0, (height - square_width) / 2.0);

        let faces = {
            let src_img = ImageRef::new(
                cap_size.0,
                cap_size.1,
                frame_data.data_u8(),
                PixelType::U8x4,
            )?;
            let pos_scale = (
                square_width / self.yolov5n_face.input_shape.0 as f32,
                square_width / self.yolov5n_face.input_shape.1 as f32,
            );
            self.yolov5n_face
                .proc_image(&src_img, 0.6, 0.5, pos_scale)?
        };

        ctx.begin_path();
        ctx.fill_paint({
            ImagePattern {
                img,
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
        ctx.rect((0.0, 0.0, width, height));
        ctx.rect(Rect {
            xy: square_xy.into(),
            size: (square_width, square_width).into(),
        });
        ctx.path_winding(WindingSolidity::Hole);
        ctx.fill_paint(nvgx::Color::rgba(1.0, 1.0, 1.0, 0.2));
        ctx.fill()?;

        ctx.save();
        ctx.stroke_paint(nvgx::Color::rgb_i(0x00, 0xBF, 0xA8));
        ctx.translate(square_xy.0, square_xy.1);
        for face in faces {
            ctx.begin_path();
            ctx.rect(face.bbox);
            ctx.stroke()?;
        }
        ctx.restore();
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
