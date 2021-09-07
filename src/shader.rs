use nalgebra::Matrix4;

use crate::raster::{Color, ScreenPt};

pub struct ShaderContext {
  pub view: Matrix4<f32>,
  pub camera: Matrix4<f32>,
  pub model: Matrix4<f32>,
}

pub trait Shader {
  fn vertex(&self, context: &ShaderContext, pt: &mut ScreenPt) {
    let matrix = context.view * context.camera * context.model;
    pt.point = matrix.transform_point(&pt.point);
  }

  fn fragment(&self, context: &ShaderContext, pt: &mut ScreenPt);
}

pub struct PureColor {
  color: Color,
}

impl PureColor {
  pub fn new(color: Color) -> Self {
    Self { color }
  }
}

impl Shader for PureColor {
  fn fragment(&self, _context: &ShaderContext, point: &mut ScreenPt) {
    point.color = self.color;
  }
}
