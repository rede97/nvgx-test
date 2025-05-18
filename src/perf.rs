use nvgx::{Align, Color, Context, Rect, RendererDevice};

#[derive(Default)]
pub struct PerfGraph<const N: usize> {
    values: Vec<f32>,
    idx: usize,
    name: String,
}

impl<const N: usize> PerfGraph<N> {
    pub fn new(name: String) -> Self {
        let mut values: Vec<f32> = Vec::with_capacity(N);
        values.resize_with(N, || 0.0);
        return Self {
            values,
            idx: 0,
            name,
        };
    }
    pub fn update(&mut self, v: f32) {
        self.values[self.idx % N] = v;
        self.idx += 1;
    }

    #[inline]
    pub fn render<R: RendererDevice, F, FTM, FTS>(
        &self,
        ctx: &mut Context<R>,
        rect: Rect,
        color: Color,
        mut val_norm: F,
        main_text: Option<FTM>,
        sub_text: Option<FTS>,
    ) -> anyhow::Result<()>
    where
        F: FnMut(f32) -> f32,
        FTM: FnOnce(f32) -> String,
        FTS: FnOnce(f32) -> String,
    {
        let average_value = self.values.iter().fold(0.0, |acc, &x| acc + x) / (N as f32);

        ctx.begin_path();

        ctx.rect(rect);
        ctx.fill_paint(nvgx::Color::rgba(0.0, 0.0, 0.0, 0.5));
        ctx.fill()?;

        ctx.begin_path();

        let bottom = rect.xy.y + rect.size.height;
        ctx.move_to((rect.xy.x, bottom));
        for (idx, v) in (0..N)
            .into_iter()
            .map(|i| (i, self.values[(i + self.idx) % N]))
        {
            let x_off = idx as f32 / (N - 1) as f32 * rect.size.width;
            let y_off = f32::clamp(val_norm(v), 0.0, 1.0) * rect.size.height;
            ctx.line_to((rect.xy.x + x_off, bottom - y_off));
        }
        ctx.line_to((rect.xy.x + rect.size.width, bottom));
        ctx.fill_paint(nvgx::Color::rgba(color.r, color.g, color.g, 0.5));
        ctx.fill()?;
        {
            ctx.text_align(Align::TOP | Align::LEFT);
            ctx.font_size(20.0);
            ctx.fill_paint(nvgx::Color::rgba_i(240, 240, 240, 192));
            ctx.text(rect.xy.offset(3.0, 3.0), &self.name)?;
        }
        if let Some(main_text) = main_text {
            ctx.text_align(Align::TOP | Align::RIGHT);
            ctx.font_size(20.0);
            ctx.fill_paint(nvgx::Color::rgba_i(240, 240, 240, 192));
            ctx.text(
                rect.xy.offset(rect.size.width - 3.0, 3.0),
                main_text(average_value),
            )?;
        }
        if let Some(sub_text) = sub_text {
            ctx.text_align(Align::BOTTOM | Align::RIGHT);
            ctx.font_size(18.0);
            ctx.fill_paint(nvgx::Color::rgba_i(240, 240, 240, 160));
            ctx.text(
                rect.xy
                    .offset(rect.size.width - 3.0, rect.size.height - 3.0),
                sub_text(average_value),
            )?;
        }

        Ok(())
    }
}
