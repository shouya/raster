use lazy_static::lazy_static;
use nalgebra::{Matrix4, Point3, Vector4};

use crate::{
  raster::{Color, Image, Pt, COLOR},
  util::reflect,
};

pub struct Light {
  // in world coordinates
  pos: Point3<f32>,
  color: Color,
}

impl Light {
  pub fn new(pos: Point3<f32>, color: Color) -> Self {
    Self { pos, color }
  }

  pub fn pos(&self) -> &Point3<f32> {
    &self.pos
  }

  pub fn color(&self) -> &Color {
    &self.color
  }
}

pub struct ShaderContext<'a> {
  pub camera: Matrix4<f32>,
  pub model: Matrix4<f32>,
  pub lights: &'a [Light],
}

pub trait Shader {
  fn vertex(&self, context: &ShaderContext, pt: &mut Pt) {
    pt.world_pos = context.model.transform_point(&pt.clip_pos);
    pt.clip_pos = context.camera.transform_point(&pt.world_pos);

    let normal = context.model.transform_vector(&pt.normal);
    pt.set_normal(normal);
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
    let light_angle = (self.light_pos - pt.world_pos).normalize();
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
  #![allow(unused)]
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
    let light_angle = (self.light_pos - pt.world_pos).normalize();
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

#[derive(Clone, Debug)]
pub struct SimpleMaterial {
  // (newmtl)
  pub name: String,
  // (Ns)
  pub specular_highlight: f32,
  // (Ks)
  pub specular_color: Color,
  // (Ka)
  pub ambient_color: Color,
  // (Kd)
  pub diffuse_color: Color,
  // (d)
  pub dissolve: f32,
  // (map_Kd)
  pub color_texture: Option<Image<Color>>,
}

impl Default for SimpleMaterial {
  fn default() -> Self {
    Self {
      name: "unnamed".into(),
      specular_highlight: 0.0,
      specular_color: COLOR::rgb(1.0, 1.0, 1.0),
      ambient_color: COLOR::rgb(1.0, 1.0, 1.0),
      diffuse_color: COLOR::rgb(1.0, 1.0, 1.0),
      dissolve: 1.0,
      color_texture: None,
    }
  }
}

impl Shader for SimpleMaterial {
  fn fragment(&self, context: &ShaderContext, pt: &mut Pt) {
    let mut color = COLOR::black();

    for light in context.lights.iter() {
      let normal = pt.normal.normalize();
      let light_angle = (light.pos() - pt.world_pos).normalize();
      let camera_angle = -context.camera.transform_point(&Point3::origin());
      let light_refl_angle = reflect(&light_angle, &normal).normalize();

      // diffuse color
      let light_intensity = light_angle.dot(&normal).max(0.0);
      let diffuse_color = light.color().component_mul(&self.diffuse_color);
      color += diffuse_color * light_intensity;

      // specular highlight color
      let specular_sharpness = light_refl_angle
        .dot(&camera_angle.coords.normalize())
        .max(0.0);
      let specular_intensity =
        f32::powf(specular_sharpness, self.specular_highlight);
      let specular_color = light.color().component_mul(&self.specular_color);

      color += specular_color * specular_intensity;
    }

    color += self.ambient_color;

    pt.color = color;
  }
}

impl SimpleMaterial {
  pub fn plaster() -> &'static Self {
    lazy_static! {
      static ref PLASTER: SimpleMaterial = SimpleMaterial {
        name: String::from("plaster"),
        specular_highlight: 0.0,
        specular_color: COLOR::rgb(1.0, 1.0, 1.0),
        ambient_color: COLOR::rgb(0.3, 0.3, 0.3),
        diffuse_color: COLOR::rgb(1.0, 1.0, 1.0),
        dissolve: 1.0,
        color_texture: None,
      };
    }

    &PLASTER
  }
}
