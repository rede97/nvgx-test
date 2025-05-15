#[macro_use]
#[allow(unused)]
extern crate anyhow;

mod demo;
use anyhow::Error;
use nvgx::*;

fn fill_size(img_size: (u32, u32), win_size: (u32, u32)) -> (f32, f32) {
    if img_size.0 >= img_size.1 {
        let ratio = img_size.1 as f32 / img_size.0 as f32;
        let height = f32::min(win_size.0 as f32 * ratio, win_size.1 as f32);
        let width = height / ratio;
        return (width, height);
    } else {
        let ratio = img_size.0 as f32 / img_size.1 as f32;
        let width = f32::min(win_size.1 as f32 * ratio, win_size.0 as f32);
        let height = width / ratio;
        return (width, height);
    }
}

struct DemoDraw {
    img_size: Option<(ImageId, (u32, u32))>,
    img: Option<ImageId>,
    camera: kamera::Camera,
}

impl<R: RendererDevice> demo::Demo<R> for DemoDraw {
    fn init(&mut self, ctx: &mut Context<R>, _scale_factor: f32) -> Result<(), Error> {
        ctx.create_font_from_file("roboto", demo::FONT_PATH)?;
        self.img = Some(
            ctx.create_image_from_file(ImageFlags::REPEATX | ImageFlags::REPEATY, demo::IMG_PATH)?,
        );
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
        self.img_size = Some(img_size);
        let fill_size = fill_size(img_size.1, (width as u32, height as u32));
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
        // ctx.fill_paint(ImagePattern {
        //     img: self.img.unwrap(),
        //     center: (0.0, 0.0).into(),
        //     size: (200.0, 200.0).into(),
        //     angle: 0.0,
        //     alpha: 0.8,
        // });
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
        ctx.stroke_paint(nvgx::Color::rgb(0.4, 0.6, 0.9));
        ctx.stroke()?;

        Ok(())
    }
}

fn main() {
    demo::run(
        DemoDraw {
            img_size: None,
            img: None,
            camera: kamera::Camera::new_default_device(),
        },
        "demo-draw",
    );
}
