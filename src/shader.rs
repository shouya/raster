use nalgebra::{Matrix4, Point3, Vector4};

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
    let normal = context.model.transform_vector(&pt.normal);
    pt.set_normal(normal);
  }

  fn fragment(&self, context: &ShaderContext, pt: &mut ScreenPt);
}

pub struct PureColor {
  color: Color,
}

impl PureColor {
  #[allow(unused)]
  pub fn new(color: Color) -> Self {
    Self { color }
  }
}

impl Shader for PureColor {
  fn fragment(&self, _context: &ShaderContext, point: &mut ScreenPt) {
    point.color = self.color;
  }
}

pub struct DiffuseShader {
  color: Color,
  light: Color,
  light_pos: Point3<f32>,
}

impl DiffuseShader {
  pub fn new(color: Color, light: Color, light_pos: Point3<f32>) -> Self {
    Self {
      color,
      light,
      light_pos,
    }
  }
}

impl Shader for DiffuseShader {
  fn fragment(&self, context: &ShaderContext, pt: &mut ScreenPt) {
    let orig_position = context.model.transform_point(&pt.orig_point);

    let light_angle = (self.light_pos - orig_position).normalize();
    let light_intensity = f32::max(light_angle.dot(&pt.normal), 0.0);
    let ambient = self.color * 0.1;
    let mut color = Vector4::zeros();
    color += self.light.component_mul(&self.color) * light_intensity + ambient;
    pt.color = color;
  }
}
