use std::rc::Rc;

use dyn_clone::DynClone;

use crate::{
  mesh::{Mesh, PolyVert},
  raster::{Color, Image, Pt, WorldMesh, COLOR},
  types::{Mat4, Vec2, Vec3, Vec4},
  util::{divw, divw3, reflect},
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

#[derive(Clone, Debug, PartialEq)]
pub struct Light {
  // in world coordinates
  pos: Vec4,
  dir: Vec3,
  color: Color,
}

impl Light {
  pub fn new(pos: Vec3, color: Color) -> Self {
    let dir = pos.normalize();
    let pos = (pos, 1.0).into();
    Self { pos, dir, color }
  }

  pub fn pos(&self) -> &Vec4 {
    &self.pos
  }

  pub fn dir(&self) -> &Vec3 {
    &self.dir
  }

  pub fn color(&self) -> &Color {
    &self.color
  }

  pub fn extrude(&self, pt: &Vec4, distance: f32) -> Vec4 {
    let pt = divw(*pt);
    let dir = (pt - self.pos).normalize();
    dir * distance + pt
  }

  pub fn to_world_mesh(&self, mesh: Rc<Mesh<PolyVert>>) -> WorldMesh<PolyVert> {
    const SCALE: f32 = 0.03;
    WorldMesh::from(mesh)
      .transformed(Mat4::from_scale(Vec3::new(SCALE, SCALE, SCALE).into()))
      .transformed(Mat4::from_translation(self.pos.into()))
      .set_shader(PureColor::new(self.color))
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
  pub model_mat4: Mat4,
  pub view_mat4: Mat4,
  pub projection_mat4: Mat4,
  // clip = projection * view * model
  pub clip_mat4: Mat4,
  pub lights: &'a [Light],
  pub options: ShaderOptions,
}

impl<'a> ShaderContext<'a> {
  pub fn get_texture(&self, handle: TextureHandle, uv: Vec2) -> Color {
    let texture = self.textures.get(handle);

    match self.options.texture_filter_mode {
      TextureFilterMode::Nearest => texture.nearest_color(uv),
      TextureFilterMode::Bilinear => texture.bilinear_color(uv),
    }
  }

  // bump texture doesn't have to be as accurate as color texture
  pub fn get_bump_texture(&self, handle: TextureHandle, uv: Vec2) -> Color {
    self.textures.get(handle).nearest_color(uv)
  }

  pub fn update_model_matrix(&mut self, matrix: Mat4) {
    self.model_mat4 = matrix;
    self.clip_mat4 = self.projection_mat4 * self.view_mat4 * matrix;
  }
}

pub trait Shader: DynClone {
  fn vertex(&self, context: &ShaderContext, pt: &mut Pt) {
    // TODO: texture coordinates perspective correction
    pt.world_pos = context.model_mat4 * pt.pos;
    pt.pos = context.clip_mat4 * pt.pos;
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
}

impl DiffuseShader {
  #[allow(unused)]
  pub fn new(color: Color) -> Self {
    Self { color }
  }
}

impl Shader for DiffuseShader {
  fn fragment(&self, context: &ShaderContext, pt: &mut Pt) {
    let ambient = self.color * 0.2;
    let mut color = ambient;

    for light in context.lights {
      let light_angle = divw3((*light.pos() - pt.world_pos).normalize());
      let light_intensity =
        f32::max(light_angle.dot(pt.normal.normalize()), 0.0);
      color += *light.color() * self.color * light_intensity;
    }

    pt.color = color;
  }
}

#[derive(Clone)]
pub struct SpecularShader {
  color: Color,
  shininess: f32,
}

impl SpecularShader {
  #![allow(unused)]
  pub fn new(color: Color) -> Self {
    let shininess = 5.0;
    Self { color, shininess }
  }
}

impl Shader for SpecularShader {
  fn fragment(&self, context: &ShaderContext, pt: &mut Pt) {
    let ambient = self.color * 0.2;
    let mut color = ambient;

    let normal = pt.normal.normalize();

    for light in context.lights {
      let light_angle = divw3((*light.pos() - pt.world_pos).normalize());
      let camera_angle = Vec3::new(0.0, 0.0, -1.0);
      let light_refl_angle = reflect(&light_angle, &normal).normalize();

      let light_intensity = f32::max(light_angle.dot(normal), 0.0);
      color += (*light.color() * self.color) * light_intensity;

      let phong_intensity =
        f32::max(light_refl_angle.dot(camera_angle.normalize()), 0.0);

      color += f32::powf(phong_intensity, self.shininess) * *light.color();
    }

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
  // (map_Bump)
  pub bump_texture: Option<TextureHandle>,
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
      bump_texture: None,
    }
  }
}

impl SimpleMaterial {
  fn diffuse_color(
    &self,
    context: &ShaderContext,
    pt: &Pt,
    light: &Light,
    normal: Vec3,
  ) -> Color {
    let light_angle = (divw3(*light.pos()) - divw3(pt.world_pos)).normalize();
    // this is
    // Vec3::from(context.camera.project_point3(Vec3::ZERO.into()))
    //   * -1.0;

    let light_intensity = light_angle.dot(normal).max(0.0);
    let diffuse_color = match self.color_texture {
      Some(handle) => context.get_texture(handle, pt.uv),
      None => self.diffuse_color,
    };
    let diffuse_light_color = *light.color() * diffuse_color;
    diffuse_light_color * light_intensity
  }

  fn specular_color(&self, pt: &Pt, light: &Light, normal: Vec3) -> Color {
    let light_angle = (divw3(*light.pos()) - divw3(pt.world_pos)).normalize();
    let light_refl_angle = reflect(&light_angle, &normal).normalize();
    let camera_angle = Vec3::new(0.0, 0.0, -1.0);

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
    let mut normal = pt.normal.normalize();
    let bump_offset = match self.bump_texture {
      Some(handle) => context.get_bump_texture(handle, pt.uv),
      None => self.diffuse_color,
    };
    normal += bump_offset.truncate();

    let diffuse = self.diffuse_color(context, pt, light, normal);
    let specular = self.specular_color(pt, light, normal);

    diffuse + specular
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
  pub fn plaster() -> SimpleMaterial {
    SimpleMaterial {
      name: String::from("plaster"),
      specular_highlight: 4.0,
      specular_color: COLOR::rgb(1.0, 1.0, 1.0),
      ambient_color: COLOR::rgb(0.3, 0.3, 0.3),
      diffuse_color: COLOR::rgb(1.0, 1.0, 1.0),
      dissolve: 1.0,
      color_texture: None,
      bump_texture: None,
    }
  }
}
