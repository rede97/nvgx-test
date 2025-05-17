use std::fmt::Debug;

use ndarray::{ArrayBase, Ix1};
use nvgx::{Point, Rect, Vector2D};

#[allow(unused)]
pub struct YoloResult {
    pub conf: f32,
    pub bbox: Rect,
    pub landmarks: [Point; 5],
}

impl Debug for YoloResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:?}", self.conf, self.bbox)
    }
}

impl YoloResult {
    pub fn new<S>(elem: ArrayBase<S, Ix1>, conf_th: f32, scale: (f32, f32)) -> Option<Self>
    where
        S: ndarray::Data<Elem = f32>,
    {
        if elem[4] > conf_th {
            // conf = obj_conf * cls_conf
            let conf = elem[4] * elem[15];
            if conf > conf_th {
                let bbox = Self::make_bbox(
                    elem[0] * scale.0,
                    elem[1] * scale.1,
                    elem[2] * scale.0,
                    elem[3] * scale.1,
                );
                return Some(Self {
                    conf,
                    bbox,
                    landmarks: [
                        (elem[5] * scale.0, elem[6] * scale.1).into(),
                        (elem[7] * scale.0, elem[8] * scale.1).into(),
                        (elem[9] * scale.0, elem[10] * scale.1).into(),
                        (elem[11] * scale.0, elem[12] * scale.1).into(),
                        (elem[13] * scale.0, elem[14] * scale.1).into(),
                    ],
                });
            }
        }
        None
    }

    #[inline]
    fn make_bbox(cx: f32, cy: f32, w: f32, h: f32) -> Rect {
        use std::ops::Sub;
        let cp: Point = (cx, cy).into();
        let s: Point = (w, h).into();
        return Rect {
            xy: cp.sub(&s.mul(0.5)),
            size: (w, h).into(),
        };
    }

    pub fn iou(&self, rhs: &Self) -> f32 {
        let i = self.bbox.intersect(rhs.bbox).area();
        if i < f32::EPSILON {
            return 0.0;
        }
        let u = (self.bbox.area() + rhs.bbox.area()) - i;
        return i / u;
    }
}
