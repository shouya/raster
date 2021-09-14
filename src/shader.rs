use nalgebra::{Matrix4, Point3, Vector4};

use crate::{
  raster::{Color, Pt, COLOR},
  util::reflect,
};

pub struct ShaderContext {
  pub camera: Matrix4<f32>,
  pub model: Matrix4<f32>,
}

pub trait Shader {
  fn vertex(&self, context: &ShaderContext, pt: &mut Pt) {
    let matrix = context.camera * context.model;
    pt.point = matrix.transform_point(&pt.point);
    let normal = context.model.transform_vector(&pt.normal);
    pt.set_normal(normal);
    pt.orig_point = context.model.transform_point(&pt.orig_point);
  }

  fn fragment(&self, context: &ShaderContext, pt: &mut Pt);
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
  fn fragment(&self, _context: &ShaderContext, point: &mut Pt) {
    point.color = self.color;
  }
}

pub struct DiffuseShader {
  color: Color,
  light: Color,
  light_pos: Point3<f32>,
}

impl DiffuseShader {
  #[allow(unused)]
  pub fn new(color: Color, light_pos: Point3<f32>) -> Self {
    let light = COLOR::rgb(1.0, 1.0, 1.0);

    Self {
      color,
      light,
      light_pos,
    }
  }
}

impl Shader for DiffuseShader {
  fn fragment(&self, _context: &ShaderContext, pt: &mut Pt) {
    let light_angle = (self.light_pos - pt.orig_point).normalize();
    let light_intensity =
      f32::max(light_angle.dot(&pt.normal.normalize()), 0.0);
    let ambient = self.color * 0.2;
    let mut color = Vector4::zeros();
    color += self.light.component_mul(&self.color) * light_intensity + ambient;
    pt.color = color;
  }
}

pub struct SpecularShader {
  light: Color,
  light_pos: Point3<f32>,
  color: Color,
  shininess: f32,
}

impl SpecularShader {
  pub fn new(color: Color, light_pos: Point3<f32>) -> Self {
    let shininess = 5.0;
    let light = COLOR::rgb(1.0, 1.0, 1.0);
    Self {
      light,
      light_pos,
      color,
      shininess,
    }
  }
}

impl Shader for SpecularShader {
  fn fragment(&self, context: &ShaderContext, pt: &mut Pt) {
    let normal = pt.normal.normalize();
    let light_angle = (self.light_pos - pt.orig_point).normalize();
    let camera_angle = -context.camera.transform_point(&Point3::origin());
    let light_refl_angle = reflect(&light_angle, &normal).normalize();
    let ambient = self.color * 0.1;
    let mut color = Vector4::zeros();

    let light_intensity = f32::max(light_angle.dot(&normal), 0.0);
    color += self.light.component_mul(&self.color) * light_intensity;

    let phong_intensity =
      f32::max(light_refl_angle.dot(&camera_angle.coords.normalize()), 0.0);

    color += f32::powf(phong_intensity, self.shininess) * self.light;

    color += ambient;

    pt.color = color;
  }
}
