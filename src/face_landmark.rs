use std::path::Path;

use fast_image_resize::{IntoImageView, ResizeOptions, Resizer, images::Image};
use ndarray::{Array4, ArrayView, IndexLonger};
use ndarray::{Axis, s};
use nvgx::Rect;
use ort::{
    execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider},
    inputs,
    session::Session,
};
use rayon::iter::{ParallelBridge, ParallelIterator};
use tracy_client::span;

use crate::utils::sigmoid;

pub struct FaceLandmarkResult {
    pub points: Vec<(f32, f32, f32)>,
    pub score: f32,
    pub tongue: f32,
}

pub struct FaceLandmark {
    session: Session,
    resizer: Resizer,
}

impl FaceLandmark {
    pub const INPUT_SIZE: usize = 256;
    pub const MARKS_NUM: usize = 478;

    pub fn new<P: AsRef<Path>>(model: P) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([
                DirectMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])?
            .with_inter_threads(4)?
            .with_parallel_execution(true)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model)
            .expect("faceland mark model");
        Ok(Self {
            session,
            resizer: Resizer::new(),
        })
    }

    pub fn proc_image(
        &mut self,
        src_image: &impl IntoImageView,
        face_rect: Rect,
        scale_to: (f32, f32),
    ) -> anyhow::Result<Option<FaceLandmarkResult>> {
        let _flm = span!("FacelandMark Face");
        _flm.emit_color(0xfe602f);
        let input_array: Array4<f32> = {
            let _preproc = span!("Pre Proc");
            let mut dst_img = Image::new(
                Self::INPUT_SIZE as u32,
                Self::INPUT_SIZE as u32,
                fast_image_resize::PixelType::U8x4,
            );

            self.resizer.resize(
                src_image,
                &mut dst_img,
                &ResizeOptions::new().crop(
                    face_rect.xy.x as f64,
                    face_rect.xy.y as f64,
                    face_rect.size.width as f64,
                    face_rect.size.height as f64,
                ),
            )?;

            let array_view = ArrayView::from_shape(
                (1, Self::INPUT_SIZE, Self::INPUT_SIZE, 4),
                dst_img.buffer(),
            )?;

            let rgb_array_view = array_view.slice(s![.., .., .., 0..3;-1]);
            let mut input_array: Array4<f32> =
                unsafe { Array4::uninit(rgb_array_view.dim()).assume_init() };
            rgb_array_view
                .axis_iter(Axis(1))
                .zip(input_array.axis_iter_mut(Axis(1)))
                .par_bridge()
                .for_each(|(old, mut new)| {
                    new.zip_mut_with(&old, |new, old| *new = *old as f32 / 255.0);
                });

            input_array
        };
        let outputs = {
            let _inference = span!("Inference");
            self.session.run(inputs![input_array.view()].unwrap())?
        };
        {
            let _post_proc = span!("Post Proc");
            let score = sigmoid(
                outputs[1]
                    .try_extract_tensor::<f32>()
                    .unwrap()
                    .index([0, 0, 0, 0]),
            );
            if score < 0.5 {
                return Ok(None);
            }
            let tongue = *(outputs[2]
                .try_extract_tensor::<f32>()
                .unwrap()
                .index([0, 0]));

            let marks = outputs[0]
                .try_extract_tensor::<f32>()
                .unwrap()
                .into_shape_with_order((Self::MARKS_NUM, 3))
                .unwrap();
            let pos_scale = (
                scale_to.0 / Self::INPUT_SIZE as f32,
                scale_to.1 / Self::INPUT_SIZE as f32,
            );

            let points: Vec<(f32, f32, f32)> = marks
                .axis_iter(Axis(0))
                .map(|p| (pos_scale.0 * p[0], pos_scale.1 * p[1], p[2]))
                .collect();
            return Ok(Some(FaceLandmarkResult {
                points,
                score,
                tongue,
            }));
        }
    }
}
