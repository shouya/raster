use std::rc::Rc;

use dyn_clone::DynClone;

use crate::{
  raster::{Color, Image, Pt, WorldMesh, COLOR},
  types::{Mat4, Point3, Vector2, Vector3, Vector4},
  util::reflect,
  wavefront::Mesh,
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
  pos: Point3,
  color: Color,
}

impl Light {
  pub fn new(pos: Point3, color: Color) -> Self {
    Self { pos, color }
  }

  pub fn pos(&self) -> &Point3 {
    &self.pos
  }

  pub fn color(&self) -> &Color {
    &self.color
  }

  pub fn project(&self, pt: &Point3, distance: f32) -> Point3 {
    let dir = (*pt - self.pos).normalize();
    dir * distance + *pt
  }

  pub fn to_world_mesh(&self, mesh: Rc<Mesh>) -> WorldMesh {
    const SCALE: f32 = 0.03;
    WorldMesh::from(mesh)
      .transformed(Mat4::from_scale(Vector3::new(SCALE, SCALE, SCALE).into()))
      .transformed(Mat4::from_translation(self.pos.into()))
      .set_shader(Rc::new(PureColor::new(self.color)))
      .double_faced(true)
      .casts_shadow(false)
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
  pub camera: Mat4,
  pub lights: &'a [Light],
  pub options: ShaderOptions,
}

impl<'a> ShaderContext<'a> {
  pub fn get_texture(&self, handle: TextureHandle, uv: Vector2) -> Color {
    let texture = self.textures.get(handle);

    match self.options.texture_filter_mode {
      TextureFilterMode::Nearest => texture.nearest_color(uv),
      TextureFilterMode::Bilinear => texture.bilinear_color(uv),
    }
  }
}

pub trait Shader: DynClone {
  fn vertex(&self, context: &ShaderContext, pt: &mut Pt) {
    pt.clip_pos = context.camera.project_point3(pt.world_pos.into()).into();
  }

  fn fragment(&self, context: &ShaderContext, pt: &mut Pt);
}

#[derive(Clone)]
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

#[derive(Clone)]
pub struct DiffuseShader {
  color: Color,
  light: Color,
  light_pos: Point3,
}

impl DiffuseShader {
  #[allow(unused)]
  pub fn new(color: Color, light_pos: Point3) -> Self {
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
    let light_intensity = f32::max(light_angle.dot(pt.normal.normalize()), 0.0);
    let ambient = self.color * 0.2;
    let mut color = Vector4::ZERO;
    color += self.light * self.color * light_intensity + ambient;
    pt.color = color;
  }
}

#[derive(Clone)]
pub struct SpecularShader {
  light: Color,
  light_pos: Point3,
  color: Color,
  shininess: f32,
}

impl SpecularShader {
  #![allow(unused)]
  pub fn new(color: Color, light_pos: Point3) -> Self {
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
    let camera_angle =
      Vector3::from(context.camera.project_point3(Point3::ZERO.into())) * -1.0;
    let light_refl_angle = reflect(&light_angle, &normal).normalize();
    let ambient = self.color * 0.1;
    let mut color = Vector4::ZERO;

    let light_intensity = f32::max(light_angle.dot(normal), 0.0);
    color += (self.light * self.color) * light_intensity;

    let phong_intensity =
      f32::max(light_refl_angle.dot(camera_angle.normalize()), 0.0);

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

impl SimpleMaterial {
  fn diffuse_color(
    &self,
    context: &ShaderContext,
    pt: &Pt,
    light: &Light,
  ) -> Color {
    let normal = pt.normal.normalize();
    let light_angle = (*light.pos() - pt.world_pos).normalize();
    // this is
    // Vector3::from(context.camera.project_point3(Point3::ZERO.into()))
    //   * -1.0;

    // diffuse color
    let light_intensity = light_angle.dot(normal).max(0.0);
    let diffuse_color = match self.color_texture {
      Some(handle) => context.get_texture(handle, pt.uv),
      None => self.diffuse_color,
    };
    let diffuse_light_color = *light.color() * diffuse_color;
    diffuse_light_color * light_intensity
  }

  fn specular_color(&self, pt: &Pt, light: &Light) -> Color {
    let normal = pt.normal.normalize();
    let light_angle = (*light.pos() - pt.world_pos).normalize();
    let light_refl_angle = reflect(&light_angle, &normal).normalize();
    let camera_angle = Vector3::new(0.0, 0.0, -1.0);

    let specular_sharpness =
      light_refl_angle.dot(camera_angle.normalize()).max(0.0);
    let specular_intensity = if self.specular_highlight > 1.0 {
      f32::powf(specular_sharpness, self.specular_highlight)
    } else {
      0.0
    };

    *light.color() * self.specular_color * specular_intensity
  }

  fn color_from_light(
    &self,
    context: &ShaderContext,
    pt: &Pt,
    light: &Light,
  ) -> Color {
    self.diffuse_color(context, pt, light) + self.specular_color(pt, light)
  }

  fn fragment_ambient(&self, _context: &ShaderContext, pt: &mut Pt) {
    pt.color = self.ambient_color * 0.1;
  }
}

impl Shader for SimpleMaterial {
  fn fragment(&self, context: &ShaderContext, pt: &mut Pt) {
    self.fragment_ambient(context, pt);

    // only pixels not in shadow get color from light.
    if !pt.in_shadow.unwrap_or(false) {
      for light in context.lights.iter() {
        pt.color += self.color_from_light(context, pt, light);
      }
    }
  }
}

impl SimpleMaterial {
  pub fn plaster() -> Rc<SimpleMaterial> {
    let material = SimpleMaterial {
      name: String::from("plaster"),
      specular_highlight: 4.0,
      specular_color: COLOR::rgb(1.0, 1.0, 1.0),
      ambient_color: COLOR::rgb(0.3, 0.3, 0.3),
      diffuse_color: COLOR::rgb(1.0, 1.0, 1.0),
      dissolve: 1.0,
      color_texture: None,
    };

    Rc::new(material)
  }
}
