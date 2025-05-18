mod demo;
mod faceland;
mod perf;
mod utils;
mod yolov5_face;

use std::time::Instant;

use anyhow::Error;
use faceland::FacelandMark;
use fast_image_resize::{PixelType, images::ImageRef};
use num_traits::AsPrimitive;
use nvgx::*;
use perf::PerfGraph;
use utils::scale_rect;
use yolov5_face::YoloV5Face;

use tracy_client::{Client, span};

#[inline]
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

#[inline]
fn mk_face_land_mark_crop_from_bbox<N: AsPrimitive<f32>>(
    bbox: Rect,
    img_size: (N, N),
    margin: f32,
) -> (Rect, Point) {
    let img_size: (f32, f32) = (img_size.0.as_(), img_size.1.as_());
    let bbox = scale_rect(bbox, img_size);
    let max_size = f32::max(bbox.size.width, bbox.size.height) * margin;
    let max_size = f32::min(max_size, f32::min(img_size.0, img_size.1));
    let half_max_size = max_size / 2.0;
    let center = (
        bbox.xy.x + bbox.size.width / 2.0,
        bbox.xy.y + bbox.size.height / 2.0,
    );

    let mut left = f32::max(0.0, center.0 - half_max_size);
    let right = left + max_size;
    if right > img_size.0 {
        left = img_size.0 - max_size;
    }
    let mut top = f32::max(0.0, center.1 - half_max_size);
    let bottom = top + max_size;
    if bottom > img_size.1 {
        top = img_size.1 - max_size;
    }

    return (
        Rect {
            xy: (left, top).into(),
            size: (max_size, max_size).into(),
        },
        center.into(),
    );
}

struct DemoDraw {
    img_size: Option<(ImageId, (u32, u32))>,
    camera: kamera::Camera,
    yolov5n_face: YoloV5Face,
    face_land_mark: FacelandMark,
    prev_time: Instant,
    frame_time_graph: PerfGraph<64>,
}

impl<R: RendererDevice> demo::Demo<R> for DemoDraw {
    fn init(&mut self, ctx: &mut Context<R>, _scale_factor: f32) -> Result<(), Error> {
        ctx.create_font_from_file("roboto", demo::FONT_PATH)?;
        self.camera.start();
        Ok(())
    }

    fn update(&mut self, width: f32, height: f32, ctx: &mut Context<R>) -> anyhow::Result<()> {
        let _update_zone = span!("Frame");
        _update_zone.emit_color(0xeeeeff);
        let frame = {
            let _camera = span!("Camera");
            let Some(frame) = self.camera.wait_for_frame() else {
                return Ok(());
            };
            frame
        };
        let frame_data = frame.data();

        let cap_size = frame.size_u32();

        let cap_size_f = (cap_size.0 as f32, cap_size.1 as f32);
        let img_display_size = padding_fit_img(cap_size_f, (width, height));
        let img_display_scale = img_display_size.0 / cap_size_f.0;
        let img_display_offset: Point = (
            (width - img_display_size.0) / 2.0,
            (height - img_display_size.1) / 2.0,
        )
            .into();

        let src_img = ImageRef::new(
            cap_size.0,
            cap_size.1,
            frame_data.data_u8(),
            PixelType::U8x4,
        )?;
        let faces = self.yolov5n_face.proc_image(&src_img, 0.6, 0.5)?;

        let face_land_marks = {
            let max_conf_face = faces.iter().max_by(|a, b| a.conf.total_cmp(&b.conf));
            if let Some(face) = max_conf_face {
                use std::ops::Add;
                let yolov5_square_width_img = f32::min(cap_size_f.0, cap_size_f.1);
                let yolov5_crop_img_size = (yolov5_square_width_img, yolov5_square_width_img);
                let yolov5_crop_img_offset = (
                    (cap_size_f.0 - yolov5_crop_img_size.0) / 2.0,
                    (cap_size_f.1 - yolov5_crop_img_size.1) / 2.0,
                );
                let (relative_img_crop, _) =
                    mk_face_land_mark_crop_from_bbox(face.bbox, yolov5_crop_img_size, 1.5);
                let abs_img_crop = Rect {
                    xy: relative_img_crop.xy.add(&yolov5_crop_img_offset.into()),
                    size: relative_img_crop.size,
                };

                let face_land_mark_display_size = (
                    relative_img_crop.size.width * img_display_scale,
                    relative_img_crop.size.height * img_display_scale,
                );
                let result = self.face_land_mark.proc_image(
                    &src_img,
                    abs_img_crop,
                    face_land_mark_display_size,
                )?;

                let display_rect = Rect {
                    xy: abs_img_crop.xy.mul(img_display_scale),
                    size: face_land_mark_display_size.into(),
                };

                result.map(|v| (v, display_rect))
            } else {
                None
            }
        };

        {
            let img = {
                let _update_img = span!("Update Img");
                _update_img.emit_color(0xff2020);
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
                img
            };

            let _draw = span!("Draw");
            _draw.emit_color(0xff20f0);
            ctx.begin_path();
            ctx.fill_paint({
                ImagePattern {
                    img,
                    center: img_display_offset.into(),
                    size: img_display_size.into(),
                    angle: 0.0,
                    alpha: 1.0,
                }
            });
            ctx.rect(Rect {
                xy: img_display_offset.into(),
                size: img_display_size.into(),
            });
            ctx.fill()?;
            ctx.reset_transform();
            ctx.translate(img_display_offset.x, img_display_offset.y);

            if false {
                // draw yolov5 area
                // yolov5_display_width = min(cap_size_f.0, cap_size_f.1) * img_display_scale
                let yolov5_display_width = f32::min(img_display_size.0, img_display_size.1);
                let yolov5_display_offset: Point = (
                    (img_display_size.0 - yolov5_display_width) / 2.0,
                    (img_display_size.1 - yolov5_display_width) / 2.0,
                )
                    .into();
                {
                    // camera yolov5 mask
                    ctx.begin_path();
                    ctx.rect((0.0, 0.0, img_display_size.0, img_display_size.1));
                    ctx.rect(Rect {
                        xy: yolov5_display_offset.into(),
                        size: (yolov5_display_width, yolov5_display_width).into(),
                    });
                    ctx.path_winding(WindingSolidity::Hole);
                    ctx.fill_paint(nvgx::Color::rgba(1.0, 1.0, 1.0, 0.2));
                    ctx.fill()?;
                }
                // draw face rect
                ctx.save();
                ctx.stroke_paint(nvgx::Color::rgb_i(0x00, 0xBF, 0xA8));
                ctx.translate(yolov5_display_offset.x, yolov5_display_offset.y);
                for face in faces {
                    ctx.begin_path();
                    ctx.rounded_rect(
                        scale_rect(face.bbox, (yolov5_display_width, yolov5_display_width)),
                        10.0,
                    );
                    ctx.stroke()?;
                }
                ctx.restore();
            }

            if let Some((face_land_marks, rect)) = face_land_marks {
                ctx.save();
                ctx.stroke_paint(nvgx::Color::rgb_i(0x20, 0xBF, 0xA8));
                ctx.begin_path();
                ctx.rounded_rect(rect, 10.0);
                ctx.stroke()?;
                ctx.fill_paint(nvgx::Color::rgb_i(0xFF, 0x64, 0x64));
                ctx.font_size(30.0);
                ctx.text(
                    rect.xy,
                    &format!(
                        "score: {:.1} tongue:{:.1}",
                        (face_land_marks.score),
                        face_land_marks.tongue
                    ),
                )?;
                ctx.translate(rect.xy.x, rect.xy.y);
                {
                    ctx.begin_path();
                    ctx.fill_paint(Color::rgba_i(0x30, 0xc8, 0xff, 0x80));
                    for point in face_land_marks.points {
                        ctx.circle((point.0, point.1), 3.0);
                    }
                    ctx.fill()?;
                }
                ctx.restore();
            }

            {
                let now = Instant::now();
                let duration = now - std::mem::replace(&mut self.prev_time, now);
                self.frame_time_graph.update(duration.as_secs_f32());
                self.frame_time_graph
                    .render(
                        ctx,
                        Rect {
                            xy: (10.0, 10.0).into(),
                            size: (200.0, 50.0).into(),
                        },
                        Color::rgb_i(0x00, 0xBF, 0xA8),
                        |v| v * 1000.0 / 100.0,
                        Some(|v| format!("{:.1} FPS", 1.0 / v)),
                        Some(|v| format!("{:.1} ms", v * 1000.0)),
                    )
                    .unwrap();
            }
        }

        Ok(())
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    Client::start();
    demo::run(
        DemoDraw {
            img_size: None,
            camera: kamera::Camera::new_default_device(),
            yolov5n_face: YoloV5Face::new("weights/yolov5n-face-relu.onnx").unwrap(),
            face_land_mark: FacelandMark::new("weights/face_landmarks_detector.onnx").unwrap(),
            prev_time: Instant::now(),
            frame_time_graph: PerfGraph::new("Frame Time".into()),
        },
        "Yolov5Face-FacelandMark",
    );
}
