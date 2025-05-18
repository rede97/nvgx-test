use fast_image_resize::images::Image;
use fast_image_resize::{IntoImageView, ResizeOptions, Resizer};
use ndarray::{Array4, ArrayView};
use ndarray::{Axis, s};
use ort::execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider};
use ort::inputs;
use ort::session::Session;
use rayon::iter::{ParallelBridge, ParallelIterator};
use result::YoloResult;
use std::path::Path;

use tracy_client::span;

mod result;

#[allow(unused)]
pub struct YoloV5Face {
    session: Session,
    resizer: Resizer,
    pub input_shape: (usize, usize),
    pub output_shape: (usize, usize, usize),
    pub pos_scale: (f32, f32),
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
            .commit_from_file(model)
            .expect("yolov5 model");

        let dims = &session.inputs[0].input_type.tensor_dimensions().unwrap()[2..];
        let input_shape = (dims[0] as usize, dims[1] as usize);
        let dims = session.outputs[0].output_type.tensor_dimensions().unwrap();
        let output_shape = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
        let pos_scale = (1.0 / input_shape.0 as f32, 1.0 / input_shape.1 as f32);

        Ok(Self {
            session,
            input_shape,
            output_shape,
            resizer: Resizer::new(),
            pos_scale,
        })
    }

    pub fn proc_image(
        &mut self,
        src_image: &impl IntoImageView,
        conf_th: f32,
        iou_th: f32,
    ) -> anyhow::Result<Vec<YoloResult>> {
        let _yolov5face = span!("Yolov5 Face");
        _yolov5face.emit_color(0x2f60fe);

        let input_array: Array4<f32> = {
            let _preproc = span!("Pre Proc");
            let mut dst_img = Image::new(
                self.input_shape.0 as u32,
                self.input_shape.1 as u32,
                fast_image_resize::PixelType::U8x4,
            );
            self.resizer.resize(
                src_image,
                &mut dst_img,
                &ResizeOptions::new().fit_into_destination(None),
            )?;

            let array_view = ArrayView::from_shape(
                (1, self.input_shape.0, self.input_shape.1, 4),
                dst_img.buffer(),
            )?
            .permuted_axes([0, 3, 1, 2]); // [1, w, h, 3] -> [1, 3, w, h]

            let rgb_array_view = array_view.slice(s![.., 0..3;-1, .., ..]);

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
            // [batch_size][4032][16{xyxy:0..4, conf:4, landmarks:5..15, cls:15}]
            let _post_proc = span!("Post Proc");
            let output = outputs[0].try_extract_tensor::<f32>().unwrap();
            let output_batch = output.slice(s![0, .., ..]);

            let mut all_results = Vec::new();
            for elem in output_batch.axis_iter(Axis(0)) {
                if let Some(e) = YoloResult::new(elem, conf_th, self.pos_scale) {
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
