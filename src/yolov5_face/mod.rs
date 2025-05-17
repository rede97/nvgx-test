use fast_image_resize::images::Image;
use fast_image_resize::{IntoImageView, ResizeOptions, Resizer};
use image::{ImageBuffer, Rgb};
use ndarray::{Array4, ArrayView, ArrayView3};
use ndarray::{Axis, s};
use ort::execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider};
use ort::inputs;
use ort::session::Session;
use result::YoloResult;
use std::ops::Mul;
use std::path::Path;

mod result;

#[allow(unused)]
pub struct YoloV5Face {
    session: Session,
    resize: Resizer,
    pub input_shape: (usize, usize),
    pub output_shape: (usize, usize, usize),
    x: usize,
}

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

impl YoloV5Face {
    pub fn new<P: AsRef<Path>>(model: P) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([
                DirectMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])?
            .with_inter_threads(4)?
            .with_parallel_execution(true)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model)?;

        let dims = &session.inputs[0].input_type.tensor_dimensions().unwrap()[2..];
        let input_shape = (dims[0] as usize, dims[1] as usize);
        let dims = session.outputs[0].output_type.tensor_dimensions().unwrap();
        let output_shape = (dims[0] as usize, dims[1] as usize, dims[2] as usize);

        Ok(Self {
            session,
            input_shape,
            output_shape,
            resize: Resizer::new(),
            x: 0,
        })
    }

    pub fn proc_image(
        &mut self,
        src_image: &impl IntoImageView,
        conf_th: f32,
        iou_th: f32,
        pos_scale: (f32, f32),
    ) -> anyhow::Result<Vec<YoloResult>> {
        let mut dst_img = Image::new(
            self.input_shape.0 as u32,
            self.input_shape.1 as u32,
            fast_image_resize::PixelType::U8x4,
        );

        let mut resizer = Resizer::new();
        resizer.resize(
            src_image,
            &mut dst_img,
            &ResizeOptions::new().fit_into_destination(None),
        )?;

        let array_view = ArrayView::from_shape(
            (1, self.input_shape.0, self.input_shape.1, 4),
            dst_img.buffer(),
        )?
        .permuted_axes([0, 3, 1, 2]); // [1, w, h, 3] -> [1, 3, w, h]

        let input_array: Array4<f32> = array_view
            .slice(s![.., 0..3;-1, .., ..])
            .map(|v| *v as f32 / 255.0); // bgr@u8 -> rgb@f32
        // input_array.axis_chunks_iter_mut(ndarray::Axis(1), 1)
        // .into_par_iter().zip(
        //     x.axis_chunks_iter(ndarray::Axis(1), 1)
        // ).for_each(|(mut out, in_chunk)| {
        //     out_chun
        // });

        {
            let outputs = self.session.run(inputs![input_array.view()].unwrap())?;
            // [batch_size][4032][16{xyxy:0..4, conf:4, landmarks:5..15, cls:15}]
            let output = &outputs[0].try_extract_tensor::<f32>().unwrap();
            let output_batch = output.slice(s![0, .., ..]);

            let mut all_results = Vec::new();
            for elem in output_batch.axis_iter(Axis(0)) {
                if let Some(e) = YoloResult::new(elem, conf_th, pos_scale) {
                    Self::nms_append(&mut all_results, e, iou_th);
                }
            }
            return Ok(all_results);
        }
    }

    #[inline]
    fn nms_append(results: &mut Vec<YoloResult>, result: YoloResult, iou_th: f32) {
        for r in results.iter_mut() {
            if result.iou(r) > iou_th {
                if result.conf > r.conf {
                    *r = result;
                }
                return;
            }
        }
        results.push(result);
    }
}
