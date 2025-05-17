use std::path::Path;
use std::sync::Arc;

use fast_image_resize::images::Image;
use fast_image_resize::{IntoImageView, ResizeOptions, Resizer};
use ndarray::s;
use ndarray::{Array4, ArrayView};
use ort::execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider};
use ort::inputs;
use ort::session::Session;

pub struct YoloV5Face {
    session: Session,
    input_shape: (usize, usize),
    resize: Resizer,
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
            // .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model)?;

        let dims = &session.inputs[0].input_type.tensor_dimensions().unwrap()[2..];
        let input_shape = (dims[0] as usize, dims[1] as usize);
        println!("{:?}", input_shape);
        for out in &session.outputs {
            println!("{:?}", out.output_type.tensor_dimensions());
        }

        Ok(Self {
            session,
            input_shape,
            resize: Resizer::new(),
        })
    }

    pub fn proc(&mut self, src_image: &impl IntoImageView) -> anyhow::Result<()> {
        let mut dst_img = Image::new(
            self.input_shape.0 as u32,
            self.input_shape.1 as u32,
            fast_image_resize::PixelType::U8x3,
        );

        self.resize.resize(
            src_image,
            &mut dst_img,
            &ResizeOptions::new().fit_into_destination(None),
        )?;

        let array_view = ArrayView::from_shape(
            (1, self.input_shape.0, self.input_shape.1, 3),
            dst_img.buffer(),
        )?
        .permuted_axes([0, 3, 1, 2]); // [1, w, h, 3] -> [1, 3, w, h]

        let input_array: Array4<f32> = array_view
            .slice(s![.., ..;-1, .., ..])
            .map(|v| *v as f32 / 255.0); // bgr@u8 -> rgb@f32
        // input_array.axis_chunks_iter_mut(ndarray::Axis(1), 1)
        // .into_par_iter().zip(
        //     x.axis_chunks_iter(ndarray::Axis(1), 1)
        // ).for_each(|(mut out, in_chunk)| {
        //     out_chun
        // });
        let outpus = self.session.run(inputs![input_array.view()].unwrap())?;

        Ok(())
    }
}
