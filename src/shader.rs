use lazy_static::lazy_static;
use nalgebra::{Matrix4, Point3, Vector2, Vector4};

use crate::{
  raster::{Color, Image, Pt, COLOR},
  util::reflect,
};

pub type TextureHandle = usize;

#[derive(Clone)]
pub struct TextureStash {
  textures: Vec<Image<Color>>,
}

impl TextureStash {
  pub fn new() -> Self {
    Self { textures: vec![] }
  }

  pub fn add(&mut self, texture: Image<Color>) -> TextureHandle {
    let handle = self.textures.len();
    self.textures.push(texture);
    handle
  }

  #[allow(unused)]
  pub fn get(&self, handle: TextureHandle) -> &Image<Color> {
    &self.textures[handle]
  }
}

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

#[derive(Debug, Clone, PartialEq)]
pub enum TextureFilterMode {
  Nearest,
  Bilinear,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShaderOptions {
  pub texture_filter_mode: TextureFilterMode,
}

impl Default for ShaderOptions {
  fn default() -> Self {
    Self {
      texture_filter_mode: TextureFilterMode::Bilinear,
    }
  }
}

pub struct ShaderContext<'a> {
  pub textures: &'a TextureStash,
  pub camera: Matrix4<f32>,
  pub model: Matrix4<f32>,
  pub lights: &'a [Light],
  pub options: ShaderOptions,
}

impl<'a> ShaderContext<'a> {
  pub fn get_texture(&self, handle: TextureHandle, uv: Vector2<f32>) -> Color {
    let texture = self.textures.get(handle);

    match self.options.texture_filter_mode {
      TextureFilterMode::Nearest => texture.nearest_color(uv),
      TextureFilterMode::Bilinear => texture.bilinear_color(uv),
    }
  }
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
  pub color_texture: Option<TextureHandle>,
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
      let diffuse_color = match self.color_texture {
        Some(handle) => context.get_texture(handle, pt.uv),
        None => self.diffuse_color,
      };
      let diffuse_light_color = light.color().component_mul(&diffuse_color);
      color += diffuse_light_color * light_intensity;

      // specular highlight color
      let specular_sharpness = light_refl_angle
        .dot(&camera_angle.coords.normalize())
        .max(0.0);
      let specular_intensity = if self.specular_highlight > 1.0 {
        f32::powf(specular_sharpness, self.specular_highlight)
      } else {
        0.0
      };

      let specular_color = light.color().component_mul(&self.specular_color);

      color += specular_color * specular_intensity;
    }

    color += self.ambient_color * 0.1;

    pt.color = color;
  }
}

impl SimpleMaterial {
  pub fn plaster() -> &'static Self {
    lazy_static! {
      static ref PLASTER: SimpleMaterial = SimpleMaterial {
        name: String::from("plaster"),
        specular_highlight: 4.0,
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
