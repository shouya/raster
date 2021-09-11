use std::{cmp::max, convert::TryInto};

use approx::abs_diff_eq;
use nalgebra::{Matrix4, Orthographic3, Point3, Vector2, Vector3, Vector4};

use crate::{
  lerp::{lerp, lerp_closed_iter, Lerp},
  shader::{DiffuseShader, Shader, ShaderContext},
  util::f32_cmp,
  wavefront::Wavefront,
};

pub type Color = Vector4<f32>;

#[allow(non_snake_case)]
pub(crate) mod COLOR {
  use super::*;

  pub const fn rgb(r: f32, g: f32, b: f32) -> Color {
    rgba(r, g, b, 1.0)
  }
  pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
    Vector4::new(r, g, b, a)
  }
}

#[derive(Debug, Clone)]
pub struct Image {
  dimension: (usize, usize),
  pixels: Vec<Color>,
}

impl Image {
  pub fn new(size: (usize, usize)) -> Self {
    let len = size.0 * size.1;
    let mut buffer = Vec::with_capacity(len);
    let clear_color = Vector4::new(0.0, 0.0, 0.0, 1.0);
    for _ in 0..len {
      buffer.push(clear_color);
    }

    Self {
      dimension: size,
      pixels: buffer,
    }
  }

  pub fn pixels(&self) -> impl Iterator<Item = &Color> {
    self.pixels.iter()
  }

  pub fn width(&self) -> usize {
    self.dimension.0
  }

  pub fn height(&self) -> usize {
    self.dimension.1
  }

  pub fn pixel(&self, coords: (i32, i32)) -> Option<&Color> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }
    let idx = coords.1 * self.width() as i32 + coords.0;
    self.pixels.get(idx as usize)
  }

  pub fn pixel_mut(&mut self, coords: (i32, i32)) -> Option<&mut Color> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }
    let idx = coords.1 * self.width() as i32 + coords.0;
    self.pixels.get_mut(idx as usize)
  }
}

#[derive(Debug, Clone)]
pub struct Zbuffer {
  dimension: (usize, usize),
  // -1: unoccupied
  // 0..inf: distance
  buffer: Vec<f32>,
}

impl Zbuffer {
  pub fn new(size: (usize, usize)) -> Self {
    let len = size.0 * size.1;
    let mut buffer = Vec::with_capacity(len);
    for _ in 0..len {
      buffer.push(10.0);
    }

    Self {
      dimension: size,
      buffer,
    }
  }

  pub fn width(&self) -> usize {
    self.dimension.0
  }

  pub fn height(&self) -> usize {
    self.dimension.1
  }

  pub fn depth(&self, coords: (i32, i32)) -> Option<&f32> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }

    let idx = coords.1 * self.width() as i32 + coords.0;
    self.buffer.get(idx as usize)
  }

  pub fn put_depth(&mut self, coords: (i32, i32), d: f32) -> Option<()> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }

    if d < 0.0 || d > 1.0 {
      return None;
    }

    let idx = coords.1 * self.width() as i32 + coords.0;
    self.buffer[idx as usize] = d;
    Some(())
  }

  pub fn to_image(&self) -> Image {
    let mut img = Image::new(self.dimension);
    for x in 0..self.width() {
      for y in 0..self.height() {
        let coords = (x as i32, y as i32);
        let d = *self.depth(coords).unwrap();
        *img.pixel_mut(coords).unwrap() = COLOR::rgb(d, d, d);
      }
    }
    img
  }
}

#[derive(Debug, Clone)]
pub struct Camera {
  // world coordinate
  inv_transform: Matrix4<f32>,
  perspective: Matrix4<f32>,
}

impl Camera {
  pub fn new_perspective(
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
  ) -> Self {
    let perspective = Matrix4::new_perspective(aspect, fovy, znear, zfar);
    let inv_transform = Matrix4::identity();
    Self {
      perspective,
      inv_transform,
    }
  }

  pub fn matrix(&self) -> Matrix4<f32> {
    self.perspective * self.inv_transform
  }

  pub fn transformd(&mut self, trans: &Matrix4<f32>) {
    self.inv_transform *= trans.pseudo_inverse(0.0001).unwrap();
  }
}

#[derive(Debug, Clone)]
pub struct Trig<T> {
  vertices: [T; 3],
}

impl<T> From<[T; 3]> for Trig<T> {
  fn from(vertices: [T; 3]) -> Self {
    Self { vertices }
  }
}

impl<'a, T> Trig<T> {
  pub fn a(&self) -> &T {
    &self.vertices[0]
  }
  pub fn b(&self) -> &T {
    &self.vertices[1]
  }
  pub fn c(&self) -> &T {
    &self.vertices[2]
  }

  #[allow(unused)]
  pub fn edges<S>(&'a self) -> impl Iterator<Item = (S, S)>
  where
    S: From<&'a T>,
  {
    vec![
      (From::from(&self.vertices[0]), From::from(&self.vertices[1])),
      (From::from(&self.vertices[1]), From::from(&self.vertices[2])),
      (From::from(&self.vertices[2]), From::from(&self.vertices[0])),
    ]
    .into_iter()
  }

  pub fn vertices<S>(&'a self) -> [S; 3]
  where
    T: Clone,
    S: From<T>,
  {
    [
      From::from(self.vertices[0].clone()),
      From::from(self.vertices[1].clone()),
      From::from(self.vertices[2].clone()),
    ]
  }

  pub fn as_ref(&self) -> Trig<&T> {
    Trig {
      vertices: [self.a(), self.b(), self.c()],
    }
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T) -> (),
  {
    f(&mut self.vertices[0]);
    f(&mut self.vertices[1]);
    f(&mut self.vertices[2]);
  }

  pub fn map<S, F>(self, f: F) -> Trig<S>
  where
    F: Fn(T) -> S,
  {
    Trig {
      vertices: self.vertices.map(|x| f(x)),
    }
  }

  pub fn convert<S>(self) -> Trig<S>
  where
    S: From<T>,
  {
    Trig {
      vertices: self.vertices.map(|x| x.into()),
    }
  }
}

#[derive(Debug, Clone)]
pub struct Face {
  vertices: Vec<PolyVert>,
}

impl Face {
  pub fn new() -> Face {
    Face {
      vertices: Vec::new(),
    }
  }
  pub fn add_vert(&mut self, v: PolyVert) {
    self.vertices.push(v);
  }

  pub fn vertices(&self) -> &[PolyVert] {
    self.vertices.as_ref()
  }
}

#[derive(Debug, Clone)]
pub struct PolyVert {
  vertex_index: usize,
  texture_index: Option<usize>,
  normal_index: Option<usize>,
}

impl PolyVert {
  pub fn new(vertex_index: usize) -> Self {
    Self {
      vertex_index,
      texture_index: None,
      normal_index: None,
    }
  }

  pub fn new_texture(vertex_index: usize, texture_index: usize) -> Self {
    Self {
      vertex_index,
      texture_index: Some(texture_index),
      normal_index: None,
    }
  }

  pub fn new_normal(vertex_index: usize, normal_index: usize) -> Self {
    Self {
      vertex_index,
      normal_index: Some(normal_index),
      texture_index: None,
    }
  }

  pub fn new_texture_normal(
    vertex_index: usize,
    texture_index: usize,
    normal_index: usize,
  ) -> Self {
    Self {
      vertex_index,
      texture_index: Some(texture_index),
      normal_index: Some(normal_index),
    }
  }
}

#[derive(Debug, Clone)]
pub struct FaceRef<'a> {
  vertices: Vec<PolyVertRef<'a>>,
}

#[derive(Debug, Clone)]
pub struct PolyVertRef<'a> {
  pub vertex: &'a Point3<f32>,
  pub texture_coords: Option<&'a Vector2<f32>>,
  pub normal: Option<&'a Vector3<f32>>,
}

impl<'a> FaceRef<'a> {
  pub fn new() -> Self {
    Self {
      vertices: Vec::new(),
    }
  }

  pub fn add_vert(&mut self, v: PolyVertRef<'a>) {
    self.vertices.push(v);
  }

  pub fn tessellate(
    &self,
  ) -> impl Iterator<Item = Trig<&PolyVertRef<'a>>> + '_ {
    assert!(self.vertices.len() >= 3);

    let mut res = Vec::new();

    let v0 = &self.vertices[0];
    let vs = self.vertices.iter().skip(1).collect::<Vec<_>>();
    for v in vs.windows(2) {
      res.push(Trig::from([v0, v[0], v[1]]));
    }

    res.into_iter()
  }

  pub fn edges(&self) -> impl Iterator<Item = (ScreenPt, ScreenPt)> + '_ {
    let n = self.vertices.len();
    (0..n).map(move |i| {
      let a = &self.vertices[i];
      let b = &self.vertices[(i + 1) % n];
      (ScreenPt::from(a), ScreenPt::from(b))
    })
  }
}

pub struct Mesh {
  transform: Matrix4<f32>,
  vertices: Vec<Point3<f32>>,
  vertex_normals: Vec<Vector3<f32>>,
  texture_coords: Vec<Vector2<f32>>,
  faces: Vec<Face>,
  shader: Box<dyn Shader>,
}

impl Mesh {
  #[allow(unused)]
  pub fn new() -> Self {
    // let shader = PureColor::new(COLOR::rgba(1.0, 0.0, 0.0, 0.0));
    let shader = DiffuseShader::new(
      COLOR::rgba(1.0, 0.0, 0.0, 0.0),
      COLOR::rgba(1.0, 1.0, 1.0, 1.0),
      Point3::new(-5.0, 10.0, 0.0),
    );

    Self {
      transform: Matrix4::identity(),
      vertices: Default::default(),
      vertex_normals: Default::default(),
      texture_coords: Default::default(),
      faces: Vec::new(),
      shader: Box::new(shader),
    }
  }

  pub fn new_wavefront(wf: Wavefront) -> Self {
    let shader = DiffuseShader::new(
      COLOR::rgba(1.0, 0.0, 0.0, 0.0),
      COLOR::rgba(1.0, 1.0, 1.0, 1.0),
      Point3::new(-5.0, 10.0, 0.0),
    );

    Self {
      transform: Matrix4::identity(),
      vertices: wf.vertices,
      vertex_normals: wf.vertex_normals,
      texture_coords: wf.texture_coords,
      faces: wf.faces,
      shader: Box::new(shader),
    }
  }

  pub fn faces(&self) -> impl Iterator<Item = FaceRef<'_>> {
    self.faces.iter().map(move |f| self.get_face(f))
  }

  pub fn get_face(&self, face: &Face) -> FaceRef<'_> {
    let mut res = FaceRef::new();
    for vert in face.vertices() {
      let vertex = &self.vertices[vert.vertex_index];
      let texture_coords = vert.texture_index.map(|i| &self.texture_coords[i]);
      let normal = vert.normal_index.map(|i| &self.vertex_normals[i]);

      res.add_vert(PolyVertRef {
        vertex,
        texture_coords,
        normal,
      })
    }
    res
  }

  pub fn transformed(mut self, transform: Matrix4<f32>) -> Self {
    self.transform = transform * self.transform;
    self
  }
}

pub struct Scene {
  camera: Camera,
  meshes: Vec<Mesh>,
}

impl Scene {
  pub fn new(camera: Camera) -> Self {
    Self {
      camera,
      meshes: vec![],
    }
  }

  pub fn add_mesh(&mut self, mesh: Mesh) {
    self.meshes.push(mesh);
  }

  pub fn iter_meshes(&self) -> impl Iterator<Item = &Mesh> + '_ {
    self.meshes.iter()
  }

  pub fn camera(&self) -> &Camera {
    &self.camera
  }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RasterizerMode {
  Wireframe,
  Shaded,
  Clipped,
}

impl Default for RasterizerMode {
  fn default() -> Self {
    RasterizerMode::Shaded
  }
}

/// A point on screen with integer xy coordinates and floating depth (z)
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct ScreenPt {
  pub point: Point3<f32>,
  pub orig_point: Point3<f32>,
  pub color: Color,
  pub normal: Vector3<f32>,
  pub uv: Vector2<f32>,
}

impl<'a> From<&PolyVertRef<'a>> for ScreenPt {
  fn from(v: &PolyVertRef<'a>) -> Self {
    let mut pt = Self::new(*v.vertex);
    if let Some(uv) = v.texture_coords {
      pt.set_uv(*uv);
    }
    if let Some(normal) = v.normal {
      pt.set_normal(*normal);
    }
    pt
  }
}

impl ScreenPt {
  pub fn new(point: Point3<f32>) -> Self {
    Self {
      point,
      orig_point: point,
      color: COLOR::rgba(1.0, 0.0, 0.0, 1.0),
      uv: Vector2::new(0.0, 0.0),
      normal: point.coords,
    }
  }

  pub fn set_uv(&mut self, uv: Vector2<f32>) {
    self.uv = uv;
  }

  pub fn set_normal(&mut self, normal: Vector3<f32>) {
    self.normal = normal.normalize();
  }

  pub fn x(&self) -> f32 {
    self.point.x
  }
  pub fn y(&self) -> f32 {
    self.point.y
  }

  pub fn depth(&self) -> f32 {
    self.point.z
  }
}

impl Lerp for ScreenPt {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    if self == other {
      return *self;
    }

    ScreenPt {
      point: lerp(t, &self.point, &other.point),
      orig_point: lerp(t, &self.orig_point, &other.orig_point),
      normal: self.normal.lerp(&other.normal, t),
      color: self.color.lerp(&other.color, t),
      uv: self.uv.lerp(&other.uv, t),
    }
  }
}

pub struct Rasterizer {
  size: (f32, f32),
  mode: RasterizerMode,
  image: Image,
  zbuffer: Zbuffer,
}

impl Rasterizer {
  pub fn new(size: (usize, usize)) -> Self {
    let image = Image::new(size);
    let zbuffer = Zbuffer::new(size);
    let mode = RasterizerMode::Shaded;
    let size = (image.width() as f32, image.height() as f32);

    Self {
      size,
      image,
      zbuffer,
      mode,
    }
  }

  pub fn set_mode(&mut self, mode: RasterizerMode) {
    self.mode = mode;
  }

  pub fn rasterize(&mut self, scene: &Scene) {
    match self.mode {
      RasterizerMode::Shaded => self.rasterize_shaded(scene),
      RasterizerMode::Clipped => self.rasterize_clipping(scene),
      RasterizerMode::Wireframe => self.rasterize_wireframe(scene),
    }
  }

  pub fn rasterize_wireframe(&mut self, scene: &Scene) {
    let camera = scene.camera();
    for mesh in scene.iter_meshes() {
      for face in mesh.faces() {
        let context = self.shader_context(camera, mesh);
        let shader = mesh.shader.as_ref();

        // TODO: each vertex fragment is computed multiple times, fix it.
        for (mut a, mut b) in face.edges() {
          mesh.shader.vertex(&context, &mut a);
          mesh.shader.vertex(&context, &mut b);
          self.draw_line(a, b, &context, shader);
        }
      }
    }
  }

  pub fn rasterize_shaded(&mut self, scene: &Scene) {
    let camera = scene.camera();
    for mesh in scene.iter_meshes() {
      let shader = mesh.shader.as_ref();
      let context = self.shader_context(camera, mesh);

      for face in mesh.faces() {
        for trig in face.tessellate() {
          let mut trig = trig.convert();
          self.shade_triangle_vertices(&mut trig, &context, shader);
          self.fill_triangle(&trig, &context, shader);
        }
      }
    }
  }

  pub fn rasterize_clipping(&mut self, scene: &Scene) {
    let camera = scene.camera();
    for mesh in scene.iter_meshes() {
      let shader = mesh.shader.as_ref();
      let context = self.shader_context(camera, mesh);

      for face in mesh.faces() {
        for trig in face.tessellate() {
          let mut trig = trig.convert();
          self.shade_triangle_vertices(&mut trig, &context, shader);
          self.draw_triangle_clipped(&trig, &context, shader);
        }
      }
    }
  }

  fn clip_triangle(&self, _trig: &Trig<ScreenPt>) -> Vec<Trig<ScreenPt>> {
    vec![&_trig]
      .into_iter()
      .flat_map(|t| self.clip_triangle_component(&t, |p| p.point.z, -1.0, -1.0))
      .flat_map(|t| self.clip_triangle_component(&t, |p| p.point.z, 1.0, 1.0))
      .flat_map(|t| self.clip_triangle_component(&t, |p| p.point.x, -1.0, -1.0))
      .flat_map(|t| self.clip_triangle_component(&t, |p| p.point.x, 1.0, 1.0))
      .flat_map(|t| self.clip_triangle_component(&t, |p| p.point.y, -1.0, -1.0))
      .flat_map(|t| self.clip_triangle_component(&t, |p| p.point.y, 1.0, 1.0))
      .collect()
  }

  fn clip_triangle_component<F>(
    &self,
    trig: &Trig<ScreenPt>,
    get_comp: F,
    lim: f32,
    sign: f32,
  ) -> Vec<Trig<ScreenPt>>
  where
    F: Fn(&ScreenPt) -> f32,
  {
    let (va, vb, vc) = (trig.a(), trig.b(), trig.c());
    let v: [f32; 3] = trig.as_ref().map(get_comp).vertices();
    let (a, b, c) = (v[0], v[1], v[2]);

    // redefine lt, gt, le, ge operator based on the sign
    let lt = |x: f32, y: f32| x * sign < y * sign;
    let ge = |x: f32, y: f32| x * sign >= y * sign;
    let in_lim = |x: f32| lt(x, lim);
    let out_lim = |x: f32| ge(x, lim);

    // case 1: all vertex within range
    if in_lim(a) && in_lim(b) && in_lim(c) {
      return vec![trig.clone()];
    }

    // case 2: all vertex out of range
    if out_lim(a) && out_lim(b) && out_lim(c) {
      // within the range; draw without clipping
      return vec![];
    }

    // case 3: two vertices out of range
    if in_lim(a) && out_lim(b) && out_lim(c) {
      let new_vb = lerp((lim - a) / (b - a), va, vb);
      let new_vc = lerp((lim - a) / (c - a), va, vc);
      return vec![[*va, new_vb, new_vc].into()];
    }
    if out_lim(a) && in_lim(b) && out_lim(c) {
      let new_va = lerp((lim - b) / (a - b), vb, va);
      let new_vc = lerp((lim - b) / (c - b), vb, vc);
      return vec![[new_va, *vb, new_vc].into()];
    }
    if out_lim(a) && out_lim(b) && in_lim(c) {
      let new_va = lerp((lim - c) / (a - c), vc, va);
      let new_vb = lerp((lim - c) / (b - c), vc, vb);
      return vec![[new_va, new_vb, *vc].into()];
    }

    // case 4: one vertex out of range
    if out_lim(a) && in_lim(b) && in_lim(c) {
      let new_vb = lerp((lim - a) / (b - a), va, vb);
      let new_vc = lerp((lim - a) / (c - a), va, vc);
      return vec![[*vb, *vc, new_vb].into(), [*vc, new_vc, new_vb].into()];
    }
    if in_lim(a) && out_lim(b) && in_lim(c) {
      let new_va = lerp((lim - b) / (a - b), vb, va);
      let new_vc = lerp((lim - b) / (c - b), vb, vc);
      return vec![[*va, *vc, new_va].into(), [*vc, new_va, new_vc].into()];
    }
    if in_lim(a) && in_lim(b) && out_lim(c) {
      let new_va = lerp((lim - c) / (a - c), vc, va);
      let new_vb = lerp((lim - c) / (b - c), vc, vb);
      return vec![[*va, *vb, new_va].into(), [*vb, new_va, new_vb].into()];
    }

    unreachable!()
  }

  // return false if the line is off-view and should be skipped
  fn clip_line(&self, a: &mut ScreenPt, b: &mut ScreenPt) -> bool {
    let get_x = |p: &ScreenPt| p.point.x;
    let get_y = |p: &ScreenPt| p.point.y;
    let get_z = |p: &ScreenPt| p.point.z;

    if !Self::clip_line_component(a, b, get_x, -1.0, 1.0) {
      return false;
    }
    if !Self::clip_line_component(a, b, get_y, -1.0, 1.0) {
      return false;
    }
    if !Self::clip_line_component(a, b, get_z, -1.0, 1.0) {
      return false;
    }

    true
  }

  // return false if the line is off-view and should be skipped
  fn clip_line_component<F>(
    a: &mut ScreenPt,
    b: &mut ScreenPt,
    get_comp: F,
    min: f32,
    max: f32,
  ) -> bool
  where
    F: Fn(&ScreenPt) -> f32,
  {
    let mut av = get_comp(a);
    let mut bv = get_comp(b);

    if av < min && bv < min {
      // both beyond min; skip this line
      return false;
    }
    if av > max && bv > max {
      // both beyond max; skip this line
      return false;
    }
    if av >= min && av <= max && bv >= min && bv <= max {
      // within the range; draw without clipping
      return true;
    }

    if av < min && bv >= min {
      // clip a on min
      let t = (min - av) / (bv - av);
      assert!((0.0..=1.0).contains(&t));
      *a = lerp(t, a, b);
    } else if av >= min && bv < min {
      // clip b on min
      let t = (min - av) / (bv - av);
      assert!((0.0..=1.0).contains(&t));
      *b = lerp(t, a, b);
    }

    // recalculate because a and b may be changed
    av = get_comp(a);
    bv = get_comp(b);

    if av > max && bv <= max {
      // clip a on max
      let t = (max - av) / (bv - av);
      assert!((0.0..=1.0).contains(&t));
      *a = lerp(t, a, b);
    } else if av <= max && bv > max {
      // clip b on max
      let t = (max - av) / (bv - av);
      assert!((0.0..=1.0).contains(&t));
      *b = lerp(t, a, b);
    }

    true
  }

  pub fn to_coords(&self, pt: &ScreenPt) -> (i32, i32) {
    let (w, h) = self.size_f32();
    let point = pt.point;
    let x = ((point.x + 1.0) / 2.0 * w).round() as i32;
    let y = (h - (point.y + 1.0) / 2.0 * h).round() as i32;
    assert!(x >= 0);
    assert!(y >= 0);
    (x, y)
  }

  pub fn to_x_coord(&self, x: f32) -> i32 {
    let (w, _h) = self.size_f32();
    ((x + 1.0) / 2.0 * w).round() as i32
  }
  pub fn to_y_coord(&self, y: f32) -> i32 {
    let (_w, h) = self.size_f32();
    (h - (y + 1.0) / 2.0 * h).round() as i32
  }

  pub fn zbuffer_image(&self) -> Image {
    self.zbuffer.to_image()
  }

  pub fn into_image(self) -> Image {
    self.image
  }

  pub fn size_f32(&self) -> (f32, f32) {
    self.size
  }

  // checks the zbuffer
  fn draw_pixel(
    &mut self,
    mut p: ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    let coords = self.to_coords(&p);

    match self.zbuffer.depth(coords) {
      None => return,
      Some(d) if p.depth() > *d => return,
      Some(_) => {
        shader.fragment(context, &mut p);
        self.put_pixel(coords, p.color);
        self.zbuffer.put_depth(coords, p.depth());
      }
    }
  }

  fn is_hidden_surface(&self, triangle: &Trig<ScreenPt>) -> bool {
    let positive_direction: Vector3<f32> = [0.0, 0.0, 1.0].into();
    let v1 = triangle.b().point - triangle.a().point;
    let v2 = triangle.c().point - triangle.a().point;
    let n = v1.cross(&v2);
    n.dot(&positive_direction) < 0.0
  }

  // do not check for zbuffer
  fn put_pixel(&mut self, coords: (i32, i32), color: Color) {
    if let Some(pixel) = self.image.pixel_mut(coords) {
      *pixel = color;
    }
  }

  fn draw_line(
    &mut self,
    mut p1: ScreenPt,
    mut p2: ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    if !self.clip_line(&mut p1, &mut p2) {
      // out of screen
      return;
    }

    let (p1x, p1y) = self.to_coords(&p1);
    let (p2x, p2y) = self.to_coords(&p2);
    let dx = p2x - p1x;
    let dy = p2y - p1y;
    let n = max(dx.abs(), dy.abs());
    for pt in lerp_closed_iter(&p1, &p2, n as usize) {
      self.draw_pixel(pt, context, shader);
    }
  }

  // fill a triangle that is flat at bottom
  fn fill_upper_triangle(
    &mut self,
    top: ScreenPt,
    bottom_left: ScreenPt,
    bottom_right: ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    let top_y = self.to_y_coord(top.y());
    let bottom_y = self.to_y_coord(bottom_left.y());
    let h = (top_y - bottom_y).abs() as usize;

    let left_pts_iter = lerp_closed_iter(&top, &bottom_left, h);
    let right_pts_iter = lerp_closed_iter(&top, &bottom_right, h);

    for (l, r) in left_pts_iter.zip(right_pts_iter) {
      self.draw_horizontal_line(l, r, context, shader)
    }
  }

  fn fill_lower_triangle(
    &mut self,
    top_left: ScreenPt,
    top_right: ScreenPt,
    bottom: ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    let top_y = self.to_y_coord(top_left.y());
    let bottom_y = self.to_y_coord(bottom.y());
    let h = (top_y - bottom_y).abs() as usize;
    // non-zero h
    let left_pts_iter = lerp_closed_iter(&top_left, &bottom, h);
    let right_pts_iter = lerp_closed_iter(&top_right, &bottom, h);

    for (l, r) in left_pts_iter.zip(right_pts_iter) {
      self.draw_horizontal_line(l, r, context, shader)
    }
  }

  fn shader_context(&self, camera: &Camera, mesh: &Mesh) -> ShaderContext {
    ShaderContext {
      camera: camera.matrix(),
      model: mesh.transform.clone(),
    }
  }

  fn shade_triangle_vertices(
    &self,
    trig: &mut Trig<ScreenPt>,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    trig.map_in_place(|pt| shader.vertex(&context, pt))
  }

  fn draw_triangle_clipped(
    &mut self,
    trig: &Trig<ScreenPt>,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    for trig in self.clip_triangle(&trig) {
      if self.is_hidden_surface(&trig) {
        return;
      }
      let pts: Vec<_> = trig.vertices().into();
      self.draw_line(pts[0], pts[1], &context, shader);
      self.draw_line(pts[1], pts[2], &context, shader);
      self.draw_line(pts[2], pts[0], &context, shader);
    }
  }

  fn fill_triangle(
    &mut self,
    trig: &Trig<ScreenPt>,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    if self.is_hidden_surface(trig) {
      return;
    }
    for trig in self.clip_triangle(&trig) {
      let mut pts: Vec<_> = trig.vertices().into();

      let [upper, lower] = Self::horizontally_split_triangle(
        pts.as_mut_slice().try_into().unwrap(),
      );

      if let Some([a, b, c]) = upper {
        self.draw_line(a, b, &context, shader);
        self.draw_line(b, c, &context, shader);
        self.draw_line(c, a, &context, shader);
        self.fill_upper_triangle(a, b, c, &context, shader);
      }

      if let Some([a, b, c]) = lower {
        self.draw_line(a, b, &context, shader);
        self.draw_line(b, c, &context, shader);
        self.draw_line(c, a, &context, shader);
        self.fill_lower_triangle(a, b, c, &context, shader);
      }
    }
  }

  fn horizontally_split_triangle(
    pts: &mut [ScreenPt; 3],
  ) -> [Option<[ScreenPt; 3]>; 2] {
    const EPS: f32 = 0.003;
    pts.sort_unstable_by(|p1, p2| f32_cmp(&p1.y(), &p2.y()));

    if abs_diff_eq!(pts[0].y(), pts[2].y(), epsilon = EPS) {
      // just a flat line
      let upper_trig = [pts[0], pts[1], pts[2]];
      return [Some(upper_trig), None];
    }

    if abs_diff_eq!(pts[0].y(), pts[1].y(), epsilon = EPS) {
      // a lower triangle
      let lower_trig = [pts[0], pts[1], pts[2]];
      return [None, Some(lower_trig)];
    }

    if abs_diff_eq!(pts[1].y(), pts[2].y(), epsilon = EPS) {
      // a lower triangle
      let upper_trig = [pts[0], pts[1], pts[2]];
      return [Some(upper_trig), None];
    }

    // a normal triangle that we need to split
    let t = (pts[1].y() - pts[0].y()) / (pts[2].y() - pts[0].y());
    let ptl = lerp(t, &pts[0], &pts[2]);
    let ptr = pts[1];

    let upper_trig = [pts[0], ptl, ptr];
    let lower_trig = [ptl, ptr, pts[2]];

    [Some(upper_trig), Some(lower_trig)]
  }

  fn draw_horizontal_line(
    &mut self,
    p1: ScreenPt,
    p2: ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    let x1 = self.to_x_coord(p1.x());
    let x2 = self.to_x_coord(p2.x());
    let w = (x1 - x2).abs() as usize;
    for p in lerp_closed_iter(&p1, &p2, w) {
      self.draw_pixel(p, context, shader);
    }
  }
}

#[cfg(test)]
mod test {
  use std::f32::consts::PI;

  use super::*;

  #[test]
  fn test_camera() {
    let aspect = 16.0 / 9.0;
    let fov = 130.0 / 180.0 * PI;
    let znear = 1.0;
    let zfar = 10.0;
    let perspective = Matrix4::new_perspective(aspect, fov, znear, zfar);
    for z in -10..10 {
      let pt = Point3::new(1.0, 2.0, z as f32);
      dbg!(z);
      dbg!(1.0 / perspective.transform_point(&pt).z);
    }

    assert!(false);
  }
}
